/* Headless verification of the browser simulator's computational core.
 *
 * The browser build re-implements the same physics as uncomputation_demo.py in
 * JavaScript. This script feeds BOTH implementations the identical problem
 * instance -- handed over in fixture.js, because the two languages' RNGs differ
 * -- and checks that they agree. Two independent implementations agreeing is
 * much stronger evidence than either one on its own.
 *
 * It also re-runs the page's own internal self-checks and the structural
 * scaling claim, so a regression in app.js fails here rather than silently in
 * someone's browser.
 *
 * Run with JavaScriptCore (ships with macOS):
 *
 *   python ../uncomputation_demo.py --no-plot --emit-js-fixture simulation/fixture.js
 *   /System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc \
 *       simulation/app.js simulation/fixture.js simulation/verify_core.js
 *
 * or with Node:  node -e "..." (see README).
 */
(function () {
  "use strict";

  var core = (typeof UncomputationCore !== "undefined")
    ? UncomputationCore
    : (typeof globalThis !== "undefined" ? globalThis.UncomputationCore : undefined);
  if (!core) {
    print("FAIL: app.js did not expose UncomputationCore -- load it first");
    throw new Error("missing core");
  }
  if (typeof FIXTURE === "undefined") {
    print("FAIL: fixture.js not loaded. Generate it with:");
    print("      python uncomputation_demo.py --no-plot " +
          "--emit-js-fixture simulation/fixture.js");
    throw new Error("missing fixture");
  }

  var TOL = 1e-9;
  var failures = 0;
  var checks = 0;

  function close(name, actual, expected, tol) {
    checks++;
    var limit = tol === undefined ? TOL : tol;
    var delta = Math.abs(actual - expected);
    if (!(delta <= limit)) {
      failures++;
      print("FAIL  " + name + "\n      js=" + actual + "  py=" + expected +
            "  |delta|=" + delta.toExponential(3) + " > " + limit);
    } else {
      print("PASS  " + name + "  (|delta| = " + delta.toExponential(2) + ")");
    }
  }

  function assert(name, condition, detail) {
    checks++;
    if (!condition) {
      failures++;
      print("FAIL  " + name + (detail ? "\n      " + detail : ""));
    } else {
      print("PASS  " + name);
    }
  }

  // --------------------------------------------------------------------
  // 0. The page's own self-checks
  // --------------------------------------------------------------------

  var self = core.runSelfChecks();
  assert("page self-checks (Bell S=1, norm, |00> restored, U != U-dagger)",
         self.ok, self.problems.join("; "));

  // --------------------------------------------------------------------
  // 1. Same problem instance in both languages
  // --------------------------------------------------------------------

  if (FIXTURE.n_data !== core.N_DATA) {
    print("SKIP: fixture n_data=" + FIXTURE.n_data + " but the browser build " +
          "uses " + core.N_DATA + "; regenerate with --n-data " + core.N_DATA);
    throw new Error("n_data mismatch");
  }

  var steps = FIXTURE.steps;
  var expected = FIXTURE.expected;
  var uniform = { kind: "uniform" };
  var basis = { kind: "basis", bits: FIXTURE.basis_bits };
  var nB = core.N_DATA + core.ANCILLAS_PER_STEP;

  print("\n--- cross-language check, N = " + FIXTURE.n_steps +
        ", seed = " + FIXTURE.seed + " ---");

  // --------------------------------------------------------------------
  // 2. Scratch register across the compute / uncompute cycle
  // --------------------------------------------------------------------

  var dirty = core.runOps(
    nB, core.uncomputedOps(steps, uniform, FIXTURE.n_steps - 1, "compute")
  );
  var clean = core.runOps(
    nB, core.uncomputedOps(steps, uniform, FIXTURE.n_steps - 1, "uncompute")
  );
  var rhoDirty = core.traceOutLeading(dirty, core.ANCILLAS_PER_STEP);
  var rhoClean = core.traceOutLeading(clean, core.ANCILLAS_PER_STEP);

  close("scratch entropy before U-dagger",
        core.vonNeumannEntropy(rhoDirty), expected.entropy_before);
  close("scratch entropy after U-dagger",
        core.vonNeumannEntropy(rhoClean), expected.entropy_after);
  close("scratch purity before U-dagger",
        core.purity(rhoDirty), expected.purity_before);
  close("scratch purity after U-dagger",
        core.purity(rhoClean), expected.purity_after);
  close("scratch trace before (must stay 1)", core.trace(rhoDirty), 1);
  close("scratch trace after (must stay 1)", core.trace(rhoClean), 1);
  assert("scratch entropy is clearly non-zero before U-dagger",
         core.vonNeumannEntropy(rhoDirty) > 0.1);

  // --------------------------------------------------------------------
  // 3. Scenario B reproduces the logical target
  // --------------------------------------------------------------------

  var full = core.runOps(nB, core.uncomputedOps(steps, uniform));
  var rhoB = core.traceOutTrailing(full, core.N_DATA);
  var model = core.structuredStates(steps, uniform);
  close("scenario B fidelity vs the logical target",
        core.fidelityPureVsMixed(model.psiRe, model.psiIm, rhoB),
        expected.fidelity_uncomputed_vs_ideal);

  // --------------------------------------------------------------------
  // 4. Scenario A: harmless on a basis input, destructive on a superposition
  // --------------------------------------------------------------------

  close("scenario A fidelity, uniform superposition input",
        core.fidelityPureVsMixed(model.psiRe, model.psiIm, model.rhoNaive),
        expected.fidelity_naive_uniform);
  close("scenario A trace distance, uniform superposition input",
        core.traceDistancePureVsMixed(model.psiRe, model.psiIm, model.rhoNaive),
        expected.trace_distance_naive_uniform);

  var basisModel = core.structuredStates(steps, basis);
  close("scenario A fidelity, computational-basis input",
        core.fidelityPureVsMixed(basisModel.psiRe, basisModel.psiIm,
                                 basisModel.rhoNaive),
        expected.fidelity_naive_basis);

  // --------------------------------------------------------------------
  // 5. JS full state vector vs JS structured model (the page's own claim)
  // --------------------------------------------------------------------

  var naiveTotal = core.N_DATA + core.ANCILLAS_PER_STEP * FIXTURE.n_steps;
  assert("measured naive width matches Python",
         naiveTotal === expected.naive_total_qubits,
         "js=" + naiveTotal + " py=" + expected.naive_total_qubits);
  assert("measured uncomputed width matches Python",
         nB === expected.uncomputed_total_qubits,
         "js=" + nB + " py=" + expected.uncomputed_total_qubits);

  if (naiveTotal <= core.MAX_SIM_QUBITS) {
    var naive = core.runOps(naiveTotal, core.naiveOps(steps, uniform));
    var rhoA = core.traceOutTrailing(naive, core.N_DATA);
    var worst = 0;
    for (var a = 0; a < rhoA.dim; a++) {
      for (var b = 0; b < rhoA.dim; b++) {
        worst = Math.max(worst,
          Math.abs(rhoA.re[a][b] - model.rhoNaive.re[a][b]),
          Math.abs(rhoA.im[a][b] - model.rhoNaive.im[a][b]));
      }
    }
    close("structured model vs full state vector (both in JS)", worst, 0);
  } else {
    print("SKIP  structured-model cross-check: " + naiveTotal +
          " qubits exceeds the browser budget of " + core.MAX_SIM_QUBITS);
  }

  // --------------------------------------------------------------------
  // 6. Structural scaling, over the slider's whole range
  // --------------------------------------------------------------------

  var scalingOk = true, detail = "";
  for (var n = 1; n <= 10; n++) {
    var generated = core.buildSteps(core.N_DATA, n, 12345);
    var naiveWires = 0, uncWires = 0, seenN = {}, seenU = {}, i, op;
    var opsA = core.naiveOps(generated, uniform);
    var opsB = core.uncomputedOps(generated, uniform);
    function tally(ops, seen) {
      for (i = 0; i < ops.length; i++) {
        op = ops[i];
        if (op.w !== undefined) seen[op.w] = 1;
        if (op.c1 !== undefined) { seen[op.c1] = 1; seen[op.c2] = 1; seen[op.t] = 1; }
      }
      return Object.keys(seen).length;
    }
    naiveWires = tally(opsA, seenN);
    uncWires = tally(opsB, seenU);
    if (naiveWires !== core.N_DATA + core.ANCILLAS_PER_STEP * n) {
      scalingOk = false;
      detail += " naive N=" + n + " touched " + naiveWires + " wires;";
    }
    if (uncWires !== core.N_DATA + core.ANCILLAS_PER_STEP) {
      scalingOk = false;
      detail += " uncomputed N=" + n + " touched " + uncWires + " wires;";
    }
  }
  assert("wires touched: O(N) naive, O(1) uncomputed, over N = 1..10",
         scalingOk, detail);

  // --------------------------------------------------------------------

  print("\n" + (checks - failures) + "/" + checks + " checks passed");
  if (failures > 0) {
    throw new Error(failures + " check(s) failed");
  }
})();
