/* Uncomputation (U-dagger) qubit-efficiency demonstration -- browser simulation.
 *
 * Everything displayed is computed here at runtime from an actual complex state
 * vector. There are no pre-computed results in this file.
 *
 * Mirrors uncomputation_demo.py:
 *   Scenario A  a fresh scratch pair per step, never cleaned  -> n_data + 2N qubits
 *   Scenario B  one scratch pair, cleaned with U-dagger        -> n_data + 2 qubits
 *
 * Wire/index convention matches PennyLane's qml.state(): wire 0 is the MOST
 * significant bit of the amplitude index.
 *
 * Plain script (not a module) so the page works when opened directly over
 * file:// -- ES modules are blocked by CORS there.
 */
(function () {
  "use strict";

  // ----------------------------------------------------------------------
  // Configuration
  // ----------------------------------------------------------------------

  var N_DATA = 4;
  var LITERALS_PER_STEP = 3;
  var ANCILLAS_PER_STEP = LITERALS_PER_STEP - 1; // k-literal AND needs k-1 scratch
  var MAX_STEPS = 10;

  // Full state-vector budget for scenario A: 2**16 amplitudes. Scenario B is
  // always n_data + 2 qubits, so it is simulated in full at every N.
  var MAX_SIM_QUBITS = 16;
  var MAX_FULL_SIM_N = Math.floor((MAX_SIM_QUBITS - N_DATA) / ANCILLAS_PER_STEP);

  var TOL = 1e-9;

  var state = { n: 4, seed: 20240517, stage: "compute" };

  // ----------------------------------------------------------------------
  // Seeded PRNG and problem specification
  // ----------------------------------------------------------------------

  /** mulberry32 -- small, seeded, reproducible. */
  function makeRng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Draw `count` reproducible steps: 3 distinct signed literals plus an angle. */
  function buildSteps(nData, count, seed) {
    if (nData < LITERALS_PER_STEP) {
      throw new Error("n_data must be >= " + LITERALS_PER_STEP);
    }
    var rng = makeRng(seed);
    var steps = [];
    for (var k = 0; k < count; k++) {
      var pool = [];
      for (var q = 0; q < nData; q++) pool.push(q);
      var qubits = [];
      for (var i = 0; i < LITERALS_PER_STEP; i++) {
        qubits.push(pool.splice(Math.floor(rng() * pool.length), 1)[0]);
      }
      var negations = [];
      for (var j = 0; j < LITERALS_PER_STEP; j++) negations.push(rng() < 0.5);
      steps.push({ qubits: qubits, negations: negations, theta: rng() * 2 * Math.PI });
    }
    return steps;
  }

  /** Classical reference: does step k's conjunction hold on basis state x? */
  function predicate(step, x, nData) {
    for (var i = 0; i < step.qubits.length; i++) {
      var bit = (x >> (nData - 1 - step.qubits[i])) & 1;
      var literal = step.negations[i] ? bit === 0 : bit === 1;
      if (!literal) return false;
    }
    return true;
  }

  /** The first Toffoli's output t = l_a AND l_b -- part of the garbage. */
  function intermediate(step, x, nData) {
    for (var i = 0; i < 2; i++) {
      var bit = (x >> (nData - 1 - step.qubits[i])) & 1;
      var literal = step.negations[i] ? bit === 0 : bit === 1;
      if (!literal) return false;
    }
    return true;
  }

  // ----------------------------------------------------------------------
  // State-vector simulator
  // ----------------------------------------------------------------------

  function zeroState(n) {
    var dim = 1 << n;
    var s = { n: n, re: new Float64Array(dim), im: new Float64Array(dim) };
    s.re[0] = 1;
    return s;
  }

  function maskFor(n, wire) {
    return 1 << (n - 1 - wire); // wire 0 is the most significant bit
  }

  function applyX(s, wire) {
    var m = maskFor(s.n, wire), dim = s.re.length, i, j, t;
    for (i = 0; i < dim; i++) {
      if ((i & m) === 0) {
        j = i | m;
        t = s.re[i]; s.re[i] = s.re[j]; s.re[j] = t;
        t = s.im[i]; s.im[i] = s.im[j]; s.im[j] = t;
      }
    }
  }

  function applyH(s, wire) {
    var m = maskFor(s.n, wire), dim = s.re.length, inv = Math.SQRT1_2, i, j;
    for (i = 0; i < dim; i++) {
      if ((i & m) === 0) {
        j = i | m;
        var ar = s.re[i], ai = s.im[i], br = s.re[j], bi = s.im[j];
        s.re[i] = (ar + br) * inv; s.im[i] = (ai + bi) * inv;
        s.re[j] = (ar - br) * inv; s.im[j] = (ai - bi) * inv;
      }
    }
  }

  function applyToffoli(s, c1, c2, target) {
    var m1 = maskFor(s.n, c1), m2 = maskFor(s.n, c2), mt = maskFor(s.n, target);
    var dim = s.re.length, i, j, t;
    for (i = 0; i < dim; i++) {
      if ((i & m1) && (i & m2) && (i & mt) === 0) {
        j = i | mt;
        t = s.re[i]; s.re[i] = s.re[j]; s.re[j] = t;
        t = s.im[i]; s.im[i] = s.im[j]; s.im[j] = t;
      }
    }
  }

  function applyPhase(s, wire, theta) {
    var m = maskFor(s.n, wire), dim = s.re.length;
    var c = Math.cos(theta), sn = Math.sin(theta);
    for (var i = 0; i < dim; i++) {
      if (i & m) {
        var r = s.re[i], im = s.im[i];
        s.re[i] = r * c - im * sn;
        s.im[i] = r * sn + im * c;
      }
    }
  }

  function applyOps(s, ops) {
    for (var i = 0; i < ops.length; i++) {
      var op = ops[i];
      if (op.g === "X") applyX(s, op.w);
      else if (op.g === "H") applyH(s, op.w);
      else if (op.g === "TOF") applyToffoli(s, op.c1, op.c2, op.t);
      else if (op.g === "PH") applyPhase(s, op.w, op.theta);
      else throw new Error("unknown gate " + op.g);
    }
  }

  /**
   * The adjoint of a gate list: reverse the order and invert each gate.
   * X, H and Toffoli are self-inverse; a phase shift inverts by negating theta.
   * This is a real U-dagger, not a second copy of U.
   */
  function adjointOps(ops) {
    var out = [];
    for (var i = ops.length - 1; i >= 0; i--) {
      var op = ops[i];
      out.push(op.g === "PH" ? { g: "PH", w: op.w, theta: -op.theta } : op);
    }
    return out;
  }

  // ----------------------------------------------------------------------
  // Circuit construction
  // ----------------------------------------------------------------------

  /** U_k: write the signed conjunction into (ancT, ancR). Not self-inverse. */
  function computeStepOps(step, ancT, ancR) {
    var ops = [], i;
    for (i = 0; i < step.qubits.length; i++) {
      if (step.negations[i]) ops.push({ g: "X", w: step.qubits[i] });
    }
    ops.push({ g: "TOF", c1: step.qubits[0], c2: step.qubits[1], t: ancT });
    ops.push({ g: "TOF", c1: ancT, c2: step.qubits[2], t: ancR });
    for (i = 0; i < step.qubits.length; i++) {
      if (step.negations[i]) ops.push({ g: "X", w: step.qubits[i] });
    }
    return ops;
  }

  function prepOps(input) {
    var ops = [], i;
    if (input.kind === "uniform") {
      for (i = 0; i < N_DATA; i++) ops.push({ g: "H", w: i });
    } else {
      for (i = 0; i < N_DATA; i++) if (input.bits[i]) ops.push({ g: "X", w: i });
    }
    return ops;
  }

  /** Scenario A: a fresh scratch pair per step, no cleanup. */
  function naiveOps(steps, input) {
    var ops = prepOps(input);
    for (var k = 0; k < steps.length; k++) {
      var ancT = N_DATA + ANCILLAS_PER_STEP * k;
      var ancR = ancT + 1;
      ops = ops.concat(computeStepOps(steps[k], ancT, ancR));
      ops.push({ g: "PH", w: ancR, theta: steps[k].theta });
    }
    return ops;
  }

  /**
   * Scenario B: one scratch pair, cleaned and reused.
   * `stopAfter` / `stopStage` halt the circuit mid-cycle so the scratch can be
   * inspected while it is still dirty.
   */
  function uncomputedOps(steps, input, stopAfter, stopStage) {
    var ancT = N_DATA, ancR = N_DATA + 1;
    var ops = prepOps(input);
    for (var k = 0; k < steps.length; k++) {
      var stepOps = computeStepOps(steps[k], ancT, ancR);
      ops = ops.concat(stepOps);
      ops.push({ g: "PH", w: ancR, theta: steps[k].theta });
      if (stopAfter !== undefined && stopAfter !== null && k === stopAfter) {
        if (stopStage === "compute") return ops;
        return ops.concat(adjointOps(stepOps));
      }
      ops = ops.concat(adjointOps(stepOps)); // U-dagger: scratch back to |0>
    }
    return ops;
  }

  function runOps(nQubits, ops) {
    var s = zeroState(nQubits);
    applyOps(s, ops);
    return s;
  }

  // ----------------------------------------------------------------------
  // Linear algebra
  // ----------------------------------------------------------------------

  function makeMatrix(dim) {
    var re = [], im = [];
    for (var i = 0; i < dim; i++) {
      re.push(new Float64Array(dim));
      im.push(new Float64Array(dim));
    }
    return { dim: dim, re: re, im: im };
  }

  /** Reduced density matrix of the LAST `k` wires (the scratch register). */
  function traceOutLeading(s, k) {
    var keepDim = 1 << k;
    var restDim = s.re.length / keepDim;
    var rho = makeMatrix(keepDim);
    for (var d = 0; d < restDim; d++) {
      var base = d * keepDim;
      for (var a = 0; a < keepDim; a++) {
        var ar = s.re[base + a], ai = s.im[base + a];
        for (var b = 0; b < keepDim; b++) {
          var br = s.re[base + b], bi = s.im[base + b];
          rho.re[a][b] += ar * br + ai * bi;   // a * conj(b)
          rho.im[a][b] += ai * br - ar * bi;
        }
      }
    }
    return rho;
  }

  /** Reduced density matrix of the FIRST `k` wires (the data register). */
  function traceOutTrailing(s, k) {
    var keepDim = 1 << k;
    var restDim = s.re.length / keepDim;
    var rho = makeMatrix(keepDim);
    for (var a = 0; a < keepDim; a++) {
      for (var b = 0; b < keepDim; b++) {
        var sr = 0, si = 0;
        for (var g = 0; g < restDim; g++) {
          var ar = s.re[a * restDim + g], ai = s.im[a * restDim + g];
          var br = s.re[b * restDim + g], bi = s.im[b * restDim + g];
          sr += ar * br + ai * bi;
          si += ai * br - ar * bi;
        }
        rho.re[a][b] = sr;
        rho.im[a][b] = si;
      }
    }
    return rho;
  }

  /** Cyclic Jacobi eigenvalues of a real symmetric matrix (array of Float64Array). */
  function jacobiEigenvalues(m, dim) {
    var a = [];
    for (var i = 0; i < dim; i++) a.push(Float64Array.from(m[i]));

    for (var sweep = 0; sweep < 100; sweep++) {
      var off = 0;
      for (var p = 0; p < dim; p++) {
        for (var q = p + 1; q < dim; q++) off += a[p][q] * a[p][q];
      }
      if (off < 1e-30) break;

      for (p = 0; p < dim; p++) {
        for (q = p + 1; q < dim; q++) {
          if (Math.abs(a[p][q]) < 1e-300) continue;
          var theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          var t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          if (theta === 0) t = 1;
          var c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (var k = 0; k < dim; k++) {
            var akp = a[k][p], akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
          }
          for (k = 0; k < dim; k++) {
            var apk = a[p][k], aqk = a[q][k];
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
          }
        }
      }
    }
    var eigs = [];
    for (i = 0; i < dim; i++) eigs.push(a[i][i]);
    return eigs.sort(function (x, y) { return x - y; });
  }

  /**
   * Eigenvalues of a complex Hermitian matrix H = A + iB, via the real
   * symmetric embedding M = [[A, -B], [B, A]]. Each eigenvalue of H appears
   * exactly twice in M, so taking every second sorted value recovers the
   * spectrum of H.
   */
  function hermitianEigenvalues(rho) {
    var d = rho.dim, big = 2 * d, m = [], i, j;
    for (i = 0; i < big; i++) m.push(new Float64Array(big));
    for (i = 0; i < d; i++) {
      for (j = 0; j < d; j++) {
        m[i][j] = rho.re[i][j];
        m[i][j + d] = -rho.im[i][j];
        m[i + d][j] = rho.im[i][j];
        m[i + d][j + d] = rho.re[i][j];
      }
    }
    var all = jacobiEigenvalues(m, big);
    var out = [];
    for (i = 0; i < big; i += 2) out.push(all[i]);
    return out;
  }

  function vonNeumannEntropy(rho) {
    var eigs = hermitianEigenvalues(rho), s = 0;
    for (var i = 0; i < eigs.length; i++) {
      // Eigenvalues are probabilities and belong in [0, 1]. Round-off can put a
      // pure state's eigenvalue at 1 + 1e-16, whose log2 is positive, which
      // would report a small NEGATIVE entropy -- an impossible quantity.
      var p = Math.min(Math.max(eigs[i], 0), 1);
      if (p > 1e-12) s -= p * Math.log2(p);
    }
    return s;
  }

  function purity(rho) {
    var p = 0;
    for (var i = 0; i < rho.dim; i++) {
      for (var j = 0; j < rho.dim; j++) {
        p += rho.re[i][j] * rho.re[i][j] + rho.im[i][j] * rho.im[i][j];
      }
    }
    return p;
  }

  function trace(rho) {
    var t = 0;
    for (var i = 0; i < rho.dim; i++) t += rho.re[i][i];
    return t;
  }

  /** F(|psi>, rho) = <psi|rho|psi>, exact when one argument is pure. */
  function fidelityPureVsMixed(psiRe, psiIm, rho) {
    var total = 0;
    for (var i = 0; i < rho.dim; i++) {
      for (var j = 0; j < rho.dim; j++) {
        // conj(psi_i) * rho_ij * psi_j, real part (the imaginary part cancels)
        var ar = psiRe[i], ai = -psiIm[i];
        var br = rho.re[i][j], bi = rho.im[i][j];
        var cr = ar * br - ai * bi, ci = ar * bi + ai * br;
        total += cr * psiRe[j] - ci * psiIm[j];
      }
    }
    return total;
  }

  function traceDistancePureVsMixed(psiRe, psiIm, rho) {
    var diff = makeMatrix(rho.dim);
    for (var i = 0; i < rho.dim; i++) {
      for (var j = 0; j < rho.dim; j++) {
        var pr = psiRe[i] * psiRe[j] + psiIm[i] * psiIm[j];
        var pi = psiIm[i] * psiRe[j] - psiRe[i] * psiIm[j];
        diff.re[i][j] = rho.re[i][j] - pr;
        diff.im[i][j] = rho.im[i][j] - pi;
      }
    }
    var eigs = hermitianEigenvalues(diff), sum = 0;
    for (i = 0; i < eigs.length; i++) sum += Math.abs(eigs[i]);
    return 0.5 * sum;
  }

  // ----------------------------------------------------------------------
  // Structured exact model for scenario A
  // ----------------------------------------------------------------------

  /**
   * Exact reduced data state for scenario A, without a full state vector.
   *
   * Valid because every compute step is a classical reversible Boolean map and
   * every use is diagonal in the computational basis, so the global state is
   * sum_x a_x e^{i phi(x)} |x>|g(x)> and coherence survives only where the
   * garbage agrees. Cross-checked against the full simulation below.
   */
  function structuredStates(steps, input) {
    var dim = 1 << N_DATA, x, k;
    var ampRe = new Float64Array(dim), ampIm = new Float64Array(dim);

    if (input.kind === "uniform") {
      var v = 1 / Math.sqrt(dim);
      for (x = 0; x < dim; x++) ampRe[x] = v;
    } else {
      var index = 0;
      for (k = 0; k < N_DATA; k++) index = (index << 1) | input.bits[k];
      ampRe[index] = 1;
    }

    var psiRe = new Float64Array(dim), psiIm = new Float64Array(dim);
    var keys = [];
    for (x = 0; x < dim; x++) {
      var phase = 0, key = "";
      for (k = 0; k < steps.length; k++) {
        var p = predicate(steps[k], x, N_DATA);
        if (p) phase += steps[k].theta;
        key += (intermediate(steps[k], x, N_DATA) ? "1" : "0") + (p ? "1" : "0");
      }
      keys.push(key);
      psiRe[x] = ampRe[x] * Math.cos(phase) - ampIm[x] * Math.sin(phase);
      psiIm[x] = ampRe[x] * Math.sin(phase) + ampIm[x] * Math.cos(phase);
    }

    var rho = makeMatrix(dim);
    for (x = 0; x < dim; x++) {
      for (var y = 0; y < dim; y++) {
        if (keys[x] !== keys[y]) continue; // garbage differs -> coherence gone
        rho.re[x][y] = psiRe[x] * psiRe[y] + psiIm[x] * psiIm[y];
        rho.im[x][y] = psiIm[x] * psiRe[y] - psiRe[x] * psiIm[y];
      }
    }
    return { psiRe: psiRe, psiIm: psiIm, rhoNaive: rho };
  }

  // ----------------------------------------------------------------------
  // Analysis for the current control settings
  // ----------------------------------------------------------------------

  function analyse(nSteps, seed, stage) {
    var steps = buildSteps(N_DATA, nSteps, seed);
    var uniform = { kind: "uniform" };
    var rng = makeRng(seed + 1);
    var bits = [];
    for (var i = 0; i < N_DATA; i++) bits.push(rng() < 0.5 ? 1 : 0);
    var basis = { kind: "basis", bits: bits };

    var out = {
      nSteps: nSteps,
      naiveAncillas: ANCILLAS_PER_STEP * nSteps,
      naiveTotal: N_DATA + ANCILLAS_PER_STEP * nSteps,
      uncAncillas: ANCILLAS_PER_STEP,
      uncTotal: N_DATA + ANCILLAS_PER_STEP,
      steps: steps
    };

    // --- Scratch register, before and after U-dagger (full simulation) ---
    var nB = N_DATA + ANCILLAS_PER_STEP;
    var dirty = runOps(nB, uncomputedOps(steps, uniform, nSteps - 1, "compute"));
    var clean = runOps(nB, uncomputedOps(steps, uniform, nSteps - 1, "uncompute"));
    var rhoDirty = traceOutLeading(dirty, ANCILLAS_PER_STEP);
    var rhoClean = traceOutLeading(clean, ANCILLAS_PER_STEP);

    out.before = {
      entropy: vonNeumannEntropy(rhoDirty),
      purity: purity(rhoDirty),
      zeroPop: rhoDirty.re[0][0],
      trace: trace(rhoDirty)
    };
    out.after = {
      entropy: vonNeumannEntropy(rhoClean),
      purity: purity(rhoClean),
      zeroPop: rhoClean.re[0][0],
      trace: trace(rhoClean)
    };
    out.stage = stage;

    // --- Logical damage, both input states -------------------------------
    out.damage = {};
    [["uniform", uniform], ["basis", basis]].forEach(function (pair) {
      var model = structuredStates(steps, pair[1]);
      out.damage[pair[0]] = {
        fidelity: fidelityPureVsMixed(model.psiRe, model.psiIm, model.rhoNaive),
        traceDistance: traceDistancePureVsMixed(model.psiRe, model.psiIm, model.rhoNaive)
      };
    });

    // --- Scenario B against the logical target ---------------------------
    var full = runOps(nB, uncomputedOps(steps, uniform));
    var rhoB = traceOutTrailing(full, N_DATA);
    var modelUniform = structuredStates(steps, uniform);
    out.uncomputedFidelity = fidelityPureVsMixed(
      modelUniform.psiRe, modelUniform.psiIm, rhoB
    );

    // --- Cross-validate the structured model where a full run is affordable
    out.crossChecked = out.naiveTotal <= MAX_SIM_QUBITS;
    if (out.crossChecked) {
      var naive = runOps(out.naiveTotal, naiveOps(steps, uniform));
      var rhoA = traceOutTrailing(naive, N_DATA);
      var worst = 0;
      for (var a = 0; a < rhoA.dim; a++) {
        for (var b = 0; b < rhoA.dim; b++) {
          worst = Math.max(
            worst,
            Math.abs(rhoA.re[a][b] - modelUniform.rhoNaive.re[a][b]),
            Math.abs(rhoA.im[a][b] - modelUniform.rhoNaive.im[a][b])
          );
        }
      }
      out.crossDeviation = worst;
    }
    return out;
  }

  // ----------------------------------------------------------------------
  // Self-checks -- run once on load, reported in the page
  // ----------------------------------------------------------------------

  function runSelfChecks() {
    var problems = [];

    // 1. A Bell state's single-qubit marginal is maximally mixed: S = 1 bit.
    var bell = zeroState(2);
    applyH(bell, 0);
    applyToffoli(bell, 0, 0, 1); // controls coincide -> acts as CNOT(0 -> 1)
    var bellRho = traceOutLeading(bell, 1);
    var bellEntropy = vonNeumannEntropy(bellRho);
    if (Math.abs(bellEntropy - 1) > 1e-6) {
      problems.push("Bell marginal entropy " + bellEntropy.toFixed(9) + " != 1");
    }

    // 2. Norm preservation through a full scenario-B circuit.
    var steps = buildSteps(N_DATA, 5, 1234);
    var s = runOps(N_DATA + 2, uncomputedOps(steps, { kind: "uniform" }));
    var norm = 0;
    for (var i = 0; i < s.re.length; i++) norm += s.re[i] * s.re[i] + s.im[i] * s.im[i];
    if (Math.abs(norm - 1) > 1e-9) problems.push("norm drift " + (norm - 1).toExponential(2));

    // 3. U-dagger must genuinely restore |0>, not merely disentangle.
    var rhoScratch = traceOutLeading(s, 2);
    if (Math.abs(rhoScratch.re[0][0] - 1) > 1e-9) {
      problems.push("scratch |00> population " + rhoScratch.re[0][0].toFixed(12));
    }

    // 4. The compute subroutine must not be self-inverse, or the demo is empty.
    var probe = buildSteps(N_DATA, 1, 99)[0];
    var forward = runOps(N_DATA + 2, prepOps({ kind: "uniform" })
      .concat(computeStepOps(probe, N_DATA, N_DATA + 1)));
    var reversed = runOps(N_DATA + 2, prepOps({ kind: "uniform" })
      .concat(adjointOps(computeStepOps(probe, N_DATA, N_DATA + 1))));
    var same = true;
    for (i = 0; i < forward.re.length; i++) {
      if (Math.abs(forward.re[i] - reversed.re[i]) > 1e-12 ||
          Math.abs(forward.im[i] - reversed.im[i]) > 1e-12) { same = false; break; }
    }
    if (same) problems.push("U equals U-dagger; the adjoint is doing nothing");

    return { ok: problems.length === 0, problems: problems, bellEntropy: bellEntropy };
  }

  // ----------------------------------------------------------------------
  // Rendering helpers
  // ----------------------------------------------------------------------

  var SVG_NS = "http://www.w3.org/2000/svg";
  function el(id) { return document.getElementById(id); }
  function fmt(v, digits) { return v.toFixed(digits === undefined ? 6 : digits); }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // ---- qubit counter tiles ----------------------------------------------

  function renderTiles(a) {
    var saved = a.naiveTotal - a.uncTotal;
    var ratio = a.naiveTotal / a.uncTotal;
    el("tiles").innerHTML =
      '<div class="tile naive"><h3>Without uncomputation</h3>' +
      '<div class="big">' + a.naiveTotal + '</div>' +
      '<div class="sub">data <span>' + N_DATA + '</span> + ancillas <span>' +
      a.naiveAncillas + '</span> &nbsp;(2 per step, all live)</div></div>' +

      '<div class="tile unc"><h3>With uncomputation</h3>' +
      '<div class="big">' + a.uncTotal + '</div>' +
      '<div class="sub">data <span>' + N_DATA + '</span> + ancillas <span>' +
      a.uncAncillas + '</span> &nbsp;(reused every step)</div></div>' +

      '<div class="tile"><h3>Qubits saved at N = ' + a.nSteps + '</h3>' +
      '<div class="big">' + saved + '</div>' +
      '<div class="sub">' + fmt(ratio, 2) + '&times; narrower circuit</div></div>';
  }

  // ---- circuit diagrams --------------------------------------------------

  function svgLine(x1, y1, x2, y2, stroke, width, opacity) {
    return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 +
      '" stroke="' + stroke + '" stroke-width="' + width + '"' +
      (opacity !== undefined ? ' opacity="' + opacity + '"' : '') + '/>';
  }

  function renderNaiveCircuit(a) {
    var rowH = 20, top = 42, labelW = 44, colW = 56, right = 26;
    var rows = N_DATA + a.naiveAncillas;
    var width = labelW + a.nSteps * colW + right;
    var height = top + rows * rowH + 14;
    var ink = cssVar("--text-secondary"), rule = cssVar("--rule");
    var muted = cssVar("--text-muted"), garbage = cssVar("--garbage");
    var parts = [];

    function y(row) { return top + row * rowH; }

    parts.push('<text x="6" y="15" font-size="10" fill="' + muted +
      '" font-family="monospace">' + rows + ' qubits allocated</text>');

    for (var r = 0; r < rows; r++) {
      var isData = r < N_DATA;
      var label = isData ? "d" + r : "a" + (r - N_DATA);
      parts.push('<text x="6" y="' + (y(r) + 3.5) + '" font-size="10" fill="' +
        (isData ? ink : muted) + '" font-family="monospace">' + label + '</text>');
      if (isData) {
        parts.push(svgLine(labelW - 8, y(r), width - 8, y(r), rule, 1.4));
      } else {
        // Scratch qubit for step k: idle until its column, garbage after it.
        var k = Math.floor((r - N_DATA) / ANCILLAS_PER_STEP);
        var xUse = labelW + k * colW + colW / 2;
        parts.push(svgLine(labelW - 8, y(r), xUse, y(r), rule, 1.4));
        parts.push(svgLine(xUse, y(r), width - 8, y(r), garbage, 2));
      }
    }

    for (var s = 0; s < a.nSteps; s++) {
      var x = labelW + s * colW + colW / 2;
      var step = a.steps[s];
      var ancT = N_DATA + ANCILLAS_PER_STEP * s, ancR = ancT + 1;
      var used = step.qubits.slice().sort(function (p, q) { return p - q; });
      parts.push(svgLine(x, y(used[0]), x, y(ancR), ink, 1.2, 0.75));
      for (var i = 0; i < used.length; i++) {
        var neg = step.negations[step.qubits.indexOf(used[i])];
        parts.push('<circle cx="' + x + '" cy="' + y(used[i]) + '" r="3.2" fill="' +
          (neg ? cssVar("--surface-1") : ink) + '" stroke="' + ink + '" stroke-width="1.2"/>');
      }
      [ancT, ancR].forEach(function (row) {
        parts.push('<circle cx="' + x + '" cy="' + y(row) + '" r="5" fill="none" stroke="' +
          garbage + '" stroke-width="1.6"/>');
        parts.push(svgLine(x - 5, y(row), x + 5, y(row), garbage, 1.6));
        parts.push(svgLine(x, y(row) - 5, x, y(row) + 5, garbage, 1.6));
      });
      parts.push('<text x="' + (x + 9) + '" y="' + (y(ancR) + 3) +
        '" font-size="9" fill="' + garbage + '" font-family="monospace">&#952;</text>');
      parts.push('<text x="' + x + '" y="' + (top - 13) + '" font-size="9" fill="' +
        muted + '" text-anchor="middle" font-family="monospace">' + (s + 1) + '</text>');
    }

    var svg = el("svg-naive");
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);
    svg.innerHTML = parts.join("");
  }

  function renderUncomputedCircuit(a) {
    var rowH = 22, top = 48, labelW = 44, colW = 128, right = 34;
    var rows = N_DATA + ANCILLAS_PER_STEP;
    var width = labelW + a.nSteps * colW + right;
    var height = top + rows * rowH + 16;
    var ink = cssVar("--text-secondary"), rule = cssVar("--rule");
    var muted = cssVar("--text-muted");
    var garbage = cssVar("--garbage"), clean = cssVar("--clean");
    var parts = [];

    function y(row) { return top + row * rowH; }

    parts.push('<text x="6" y="15" font-size="10" fill="' + muted +
      '" font-family="monospace">' + rows + ' qubits allocated, for any N</text>');

    for (var r = 0; r < rows; r++) {
      var isData = r < N_DATA;
      parts.push('<text x="6" y="' + (y(r) + 3.5) + '" font-size="10" fill="' +
        (isData ? ink : clean) + '" font-family="monospace">' +
        (isData ? "d" + r : "a" + (r - N_DATA)) + '</text>');
      parts.push(svgLine(labelW - 8, y(r), width - 8, y(r), rule, 1.4));
    }

    var ancT = N_DATA, ancR = N_DATA + 1;
    for (var s = 0; s < a.nSteps; s++) {
      var x0 = labelW + s * colW;
      var xU = x0 + 34, xTheta = x0 + 72, xAdj = x0 + 104;
      var step = a.steps[s];
      var used = step.qubits.slice().sort(function (p, q) { return p - q; });

      // Scratch is garbage only between U and U-dagger.
      parts.push(svgLine(xU, y(ancT), xAdj, y(ancT), garbage, 2.6));
      parts.push(svgLine(xU, y(ancR), xAdj, y(ancR), garbage, 2.6));

      parts.push(svgLine(xU, y(used[0]), xU, y(ancR), ink, 1.2, 0.75));
      for (var i = 0; i < used.length; i++) {
        parts.push('<circle cx="' + xU + '" cy="' + y(used[i]) + '" r="3.2" fill="' +
          ink + '"/>');
      }

      parts.push('<rect x="' + (xU - 11) + '" y="' + (y(ancT) - 9) + '" width="22" height="' +
        (rowH + 18) + '" rx="4" fill="' + cssVar("--surface-1") + '" stroke="' + garbage +
        '" stroke-width="1.6"/>');
      parts.push('<text x="' + xU + '" y="' + (y(ancT) + rowH / 2 + 4) +
        '" font-size="11" fill="' + garbage +
        '" text-anchor="middle" font-family="monospace">U</text>');

      parts.push('<rect x="' + (xTheta - 8) + '" y="' + (y(ancR) - 8) +
        '" width="16" height="16" rx="3" fill="' + cssVar("--surface-1") +
        '" stroke="' + ink + '" stroke-width="1.2"/>');
      parts.push('<text x="' + xTheta + '" y="' + (y(ancR) + 4) +
        '" font-size="10" fill="' + ink +
        '" text-anchor="middle" font-family="monospace">&#952;</text>');

      parts.push('<rect x="' + (xAdj - 13) + '" y="' + (y(ancT) - 9) + '" width="26" height="' +
        (rowH + 18) + '" rx="4" fill="' + cssVar("--surface-1") + '" stroke="' + clean +
        '" stroke-width="1.6"/>');
      parts.push('<text x="' + xAdj + '" y="' + (y(ancT) + rowH / 2 + 4) +
        '" font-size="11" fill="' + clean +
        '" text-anchor="middle" font-family="monospace">U&#8224;</text>');

      // |0> markers: the scratch is clean entering and leaving every cycle.
      // The leading marker is drawn once; after that, each step's trailing
      // marker is the next step's leading one -- which is the point.
      [ancT, ancR].forEach(function (row) {
        if (s === 0) {
          parts.push('<text x="' + (x0 + 2) + '" y="' + (y(row) - 5) +
            '" font-size="8.5" fill="' + clean +
            '" font-family="monospace">|0&#10217;</text>');
        }
        parts.push('<text x="' + (xAdj + 16) + '" y="' + (y(row) - 5) +
          '" font-size="8.5" fill="' + clean +
          '" font-family="monospace">|0&#10217;</text>');
      });

      parts.push('<text x="' + (x0 + colW / 2) + '" y="' + (top - 15) +
        '" font-size="9" fill="' + muted + '" text-anchor="middle" font-family="monospace">' +
        'step ' + (s + 1) + '</text>');
    }

    var svg = el("svg-unc");
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);
    svg.innerHTML = parts.join("");
  }

  // ---- memory chart ------------------------------------------------------

  var chartGeometry = null;

  function renderChart(current) {
    var wrap = el("chart-wrap");
    var width = Math.max(560, Math.min(wrap.clientWidth || 900, 1180));
    var height = 300;
    var m = { top: 18, right: 78, bottom: 42, left: 52 };
    var innerW = width - m.left - m.right, innerH = height - m.top - m.bottom;

    var naive = [], unc = [];
    for (var n = 1; n <= MAX_STEPS; n++) {
      naive.push(N_DATA + ANCILLAS_PER_STEP * n);
      unc.push(N_DATA + ANCILLAS_PER_STEP);
    }
    var yMax = Math.ceil((naive[naive.length - 1] * 1.12) / 4) * 4;

    function px(n) { return m.left + ((n - 1) / (MAX_STEPS - 1)) * innerW; }
    function py(v) { return m.top + innerH - (v / yMax) * innerH; }

    var ink = cssVar("--text-secondary"), muted = cssVar("--text-muted");
    var rule = cssVar("--rule"), surface = cssVar("--surface-1");
    var garbage = cssVar("--garbage"), clean = cssVar("--clean");
    var parts = [], i;

    for (var g = 0; g <= yMax; g += 4) {
      parts.push(svgLine(m.left, py(g), m.left + innerW, py(g), rule, 1));
      parts.push('<text x="' + (m.left - 9) + '" y="' + (py(g) + 3.5) +
        '" font-size="10.5" fill="' + muted +
        '" text-anchor="end" font-family="monospace">' + g + '</text>');
    }
    for (i = 1; i <= MAX_STEPS; i++) {
      parts.push('<text x="' + px(i) + '" y="' + (m.top + innerH + 18) +
        '" font-size="10.5" fill="' + muted +
        '" text-anchor="middle" font-family="monospace">' + i + '</text>');
    }
    parts.push('<text x="' + (m.left + innerW / 2) + '" y="' + (height - 6) +
      '" font-size="11.5" fill="' + ink + '" text-anchor="middle">Number of operations N</text>');
    parts.push('<text transform="translate(13,' + (m.top + innerH / 2) +
      ') rotate(-90)" font-size="11.5" fill="' + ink +
      '" text-anchor="middle">Physical qubits required</text>');

    // Crosshair at the current N.
    parts.push(svgLine(px(current), m.top, px(current), m.top + innerH, ink, 1, 0.28));

    function polyline(values, colour) {
      var pts = [];
      for (var j = 0; j < values.length; j++) pts.push(px(j + 1) + "," + py(values[j]));
      return '<polyline points="' + pts.join(" ") + '" fill="none" stroke="' + colour +
        '" stroke-width="2" stroke-linejoin="round"/>';
    }
    parts.push(polyline(naive, garbage));
    parts.push(polyline(unc, clean));

    for (i = 0; i < MAX_STEPS; i++) {
      var isCurrent = i + 1 === current;
      parts.push('<circle cx="' + px(i + 1) + '" cy="' + py(naive[i]) + '" r="' +
        (isCurrent ? 5.5 : 4) + '" fill="' + garbage + '" stroke="' + surface +
        '" stroke-width="2"/>');
      parts.push('<circle cx="' + px(i + 1) + '" cy="' + py(unc[i]) + '" r="' +
        (isCurrent ? 5.5 : 4) + '" fill="' + clean + '" stroke="' + surface +
        '" stroke-width="2"/>');
    }

    // Direct labels at the line ends -- identity is never colour-alone.
    parts.push('<text x="' + (px(MAX_STEPS) + 9) + '" y="' + (py(naive[MAX_STEPS - 1]) + 4) +
      '" font-size="11" fill="' + garbage + '" font-family="monospace">' +
      naive[MAX_STEPS - 1] + ' qubits</text>');
    parts.push('<text x="' + (px(MAX_STEPS) + 9) + '" y="' + (py(unc[MAX_STEPS - 1]) + 4) +
      '" font-size="11" fill="' + clean + '" font-family="monospace">' +
      unc[MAX_STEPS - 1] + ' qubits</text>');

    var svg = el("chart");
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);
    svg.innerHTML = parts.join("");
    chartGeometry = { px: px, py: py, naive: naive, unc: unc, m: m, innerW: innerW };

    var rowsHtml = "<tr><th>N</th><th>Without uncomputation</th>" +
      "<th>With uncomputation</th><th>Saved</th></tr>";
    for (i = 0; i < MAX_STEPS; i++) {
      rowsHtml += "<tr><td>" + (i + 1) + "</td><td>" + naive[i] + "</td><td>" +
        unc[i] + "</td><td>" + (naive[i] - unc[i]) + "</td></tr>";
    }
    el("chart-table").innerHTML = rowsHtml;
  }

  function attachChartHover() {
    var wrap = el("chart-wrap"), tip = el("tooltip");
    wrap.addEventListener("mousemove", function (event) {
      if (!chartGeometry) return;
      var rect = el("chart").getBoundingClientRect();
      var x = event.clientX - rect.left;
      var frac = (x - chartGeometry.m.left) / chartGeometry.innerW;
      var n = Math.round(frac * (MAX_STEPS - 1)) + 1;
      if (n < 1 || n > MAX_STEPS) { tip.style.opacity = 0; return; }
      var a = chartGeometry.naive[n - 1], b = chartGeometry.unc[n - 1];
      tip.innerHTML = "N = " + n + "<br>without: " + a + " qubits<br>with: " + b +
        " qubits<br>saved: " + (a - b);
      tip.style.opacity = 1;
      tip.style.left = Math.min(chartGeometry.px(n) + 14, wrap.clientWidth - 150) + "px";
      tip.style.top = (chartGeometry.py(a) - 10) + "px";
    });
    wrap.addEventListener("mouseleave", function () { tip.style.opacity = 0; });
  }

  // ---- ancilla inspector -------------------------------------------------

  function metric(key, value, meterFrac, colour) {
    var html = '<div class="metric"><span class="k">' + key +
      '</span><span class="v">' + value + '</span></div>';
    if (meterFrac !== undefined) {
      html += '<div class="meter"><i style="width:' +
        Math.max(0, Math.min(100, meterFrac * 100)).toFixed(1) +
        '%;background:' + colour + '"></i></div>';
    }
    return html;
  }

  function renderInspector(a) {
    var maxEntropy = ANCILLAS_PER_STEP; // 2 scratch qubits -> at most 2 bits
    var garbage = cssVar("--garbage"), clean = cssVar("--clean");

    el("panel-before").innerHTML =
      '<h3>Before U&dagger; <span class="badge bad">entangled</span></h3>' +
      '<p class="stage">scratch after compute + phase, step ' + a.nSteps + '</p>' +
      metric("State", "entangled with data") +
      metric("Von Neumann entropy S", fmt(a.before.entropy) + " bits",
             a.before.entropy / maxEntropy, garbage) +
      metric("Purity Tr(&rho;&sup2;)", fmt(a.before.purity),
             a.before.purity, garbage) +
      metric("Population in |00&rang;", fmt(a.before.zeroPop),
             a.before.zeroPop, garbage) +
      metric("Reusable?", "no &mdash; would corrupt data");

    el("panel-after").innerHTML =
      '<h3>After U&dagger; <span class="badge good">|00&rang;</span></h3>' +
      '<p class="stage">scratch after the inverse, step ' + a.nSteps + '</p>' +
      metric("State", "product state |00&rang;") +
      metric("Von Neumann entropy S", a.after.entropy.toExponential(2) + " bits",
             a.after.entropy / maxEntropy, clean) +
      metric("Purity Tr(&rho;&sup2;)", fmt(a.after.purity), a.after.purity, clean) +
      metric("Population in |00&rang;", fmt(a.after.zeroPop), a.after.zeroPop, clean) +
      metric("Reusable?", "yes &mdash; next step reuses it");

    var stages = ["|0&rang;", "compute U", "entangled", "apply U&dagger;", "|0&rang;", "reuse"];
    var activeIndex = a.stage === "compute" ? 2 : 4;
    var html = "";
    for (var i = 0; i < stages.length; i++) {
      var cls = "";
      if (i === activeIndex) cls = a.stage === "compute" ? " class=\"on-bad\"" : " class=\"on\"";
      html += "<b" + cls + ">" + stages[i] + "</b>";
      if (i < stages.length - 1) html += "<i>&rarr;</i>";
    }
    el("lifecycle").innerHTML = html;
  }

  // ---- logical damage ----------------------------------------------------

  function renderDamage(a) {
    var u = a.damage.uniform, b = a.damage.basis;
    var html =
      "<tr><th>Input state</th><th>Scenario</th><th>Fidelity vs target</th>" +
      "<th>Trace distance</th><th>Verdict</th></tr>" +
      "<tr><td>uniform superposition</td><td>without uncomputation</td><td>" +
      fmt(u.fidelity) + "</td><td>" + fmt(u.traceDistance) +
      "</td><td>coherence destroyed</td></tr>" +
      "<tr><td>uniform superposition</td><td>with uncomputation</td><td>" +
      fmt(a.uncomputedFidelity) + "</td><td>&lt; 1e-9</td><td>exact</td></tr>" +
      "<tr><td>computational basis</td><td>without uncomputation</td><td>" +
      fmt(b.fidelity) + "</td><td>" + fmt(b.traceDistance) +
      "</td><td>garbage harmless here</td></tr>";
    el("damage-table").innerHTML = html;

    var note = el("selfcheck");
    var extra = a.crossChecked
      ? " | scenario A cross-check at N=" + a.nSteps +
        ": full state vector vs structured model, max deviation " +
        a.crossDeviation.toExponential(2)
      : " | scenario A at N=" + a.nSteps + " (" + a.naiveTotal +
        " qubits) exceeds the browser budget; structured model only";
    note.textContent = note.dataset.base + extra;
  }

  // ----------------------------------------------------------------------
  // Wiring
  // ----------------------------------------------------------------------

  function update() {
    var a;
    try {
      a = analyse(state.n, state.seed, state.stage);
    } catch (error) {
      el("selfcheck").className = "fail";
      el("selfcheck").textContent = "simulation failed: " + error.message;
      return;
    }
    el("steps-out").textContent = state.n;
    renderTiles(a);
    renderNaiveCircuit(a);
    renderUncomputedCircuit(a);
    renderChart(state.n);
    renderInspector(a);
    renderDamage(a);
  }

  function init() {
    el("budget-note").textContent =
      "N <= " + MAX_FULL_SIM_N + ", i.e. " + MAX_SIM_QUBITS + " qubits";

    var checks = runSelfChecks();
    var note = el("selfcheck");
    note.className = checks.ok ? "ok" : "fail";
    note.dataset.base = checks.ok
      ? "internal checks passed: Bell marginal S = " + checks.bellEntropy.toFixed(9) +
        " bit, norm preserved, U-dagger restores |00>, U != U-dagger"
      : "INTERNAL CHECK FAILED: " + checks.problems.join("; ");

    el("steps").addEventListener("input", function (event) {
      state.n = parseInt(event.target.value, 10);
      update();
    });
    el("seed").addEventListener("change", function (event) {
      var value = parseInt(event.target.value, 10);
      state.seed = isNaN(value) ? 0 : value;
      update();
    });
    Array.prototype.forEach.call(
      el("stage-toggle").querySelectorAll("button"),
      function (button) {
        button.addEventListener("click", function () {
          state.stage = button.dataset.stage;
          Array.prototype.forEach.call(
            el("stage-toggle").querySelectorAll("button"),
            function (other) {
              other.setAttribute("aria-pressed", other === button ? "true" : "false");
            }
          );
          update();
        });
      }
    );
    window.addEventListener("resize", function () { renderChart(state.n); });
    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      if (mq.addEventListener) mq.addEventListener("change", update);
    }

    attachChartHover();
    update();
  }

  // ----------------------------------------------------------------------
  // Test seam
  //
  // The computational core is exposed so it can be exercised headlessly and
  // cross-checked against the Python benchmark (see verify_core.js). The DOM
  // wiring only runs when there is a document, so loading this file in a bare
  // JS engine does not throw.
  // ----------------------------------------------------------------------

  var core = {
    N_DATA: N_DATA,
    ANCILLAS_PER_STEP: ANCILLAS_PER_STEP,
    MAX_SIM_QUBITS: MAX_SIM_QUBITS,
    buildSteps: buildSteps,
    predicate: predicate,
    computeStepOps: computeStepOps,
    adjointOps: adjointOps,
    naiveOps: naiveOps,
    uncomputedOps: uncomputedOps,
    runOps: runOps,
    traceOutLeading: traceOutLeading,
    traceOutTrailing: traceOutTrailing,
    vonNeumannEntropy: vonNeumannEntropy,
    purity: purity,
    trace: trace,
    fidelityPureVsMixed: fidelityPureVsMixed,
    traceDistancePureVsMixed: traceDistancePureVsMixed,
    structuredStates: structuredStates,
    analyse: analyse,
    runSelfChecks: runSelfChecks
  };
  if (typeof module !== "undefined" && module.exports) module.exports = core;
  else if (typeof globalThis !== "undefined") globalThis.UncomputationCore = core;
  else if (typeof this !== "undefined") this.UncomputationCore = core;

  if (typeof document === "undefined") return;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
