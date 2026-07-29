#!/usr/bin/env python3
"""QAOA for staff shift scheduling, with and without uncomputation.

Why this file exists
--------------------
``uncomputation_demo.py`` shows that ``U-dagger`` turns an ``O(N)`` scratch
requirement into ``O(1)`` on a synthetic phase oracle. This file applies the same
technique to a real constrained-optimisation problem and asks a sharper question:

    does keeping the garbage merely cost qubits, or does it actually degrade the
    quality of the solutions QAOA returns?

The problem
-----------
Staff shift scheduling. ``n_staff`` people, ``n_shifts`` shifts, one binary
variable per (staff, shift) pair: "does this person work this shift?".

Two families of hard constraint, both naturally **three-literal** -- which is the
whole reason ancillas are needed. Two-local penalty terms compile to CNOT ladders
and need no scratch at all; it is the k>=3 clauses that force it.

* **coverage**   -- every shift needs at least one person on it
                    ``(x_1s OR x_2s OR ... )``
* **no-burnout** -- nobody works every shift
                    ``NOT (x_n1 AND x_n2 AND ... )``

A clause is *violated* exactly when a conjunction of three signed literals holds,
which is precisely the subroutine already implemented and tested in
``uncomputation_demo.compute_step``. Soft preference costs are one-local and are
applied as single-qubit phases.

QAOA
----
Standard form (Farhi, Goldstone & Gutmann, arXiv:1411.4028):

    |psi(gamma, beta)> = prod_l [ U_B(beta_l) U_C(gamma_l) ] H^{tensor n} |0>

with ``U_C(gamma) = exp(-i gamma C(x))`` and ``U_B(beta) = prod_j RX(2 beta_j)``.
``(gamma, beta)`` are optimised classically.

The cost layer is where uncomputation lives. Per three-literal clause:

    A (naive)        compute violation into a FRESH scratch pair, phase, leave it
    B (uncomputed)   compute, phase, adjoint(compute), reuse the SAME pair

Simulation
----------
Scenario B is ``n_vars + 2`` qubits: full state vector, any depth.

Scenario A is ``n_vars + 2 * n_clauses`` qubits, which grows out of reach. But the
garbage written by layer ``l`` is a function of the data register at layer ``l``
and is never touched again, so it can be traced out immediately. That makes the
naive cost layer an exact **dephasing channel** on the data density matrix:

    rho  ->  D ( sum_c P_c rho P_c ) D^dagger ,   D = diag(e^{-i gamma C(x)})

where ``P_c`` projects onto a garbage class. This is exact at any depth and costs
``2**n_vars`` squared memory. It is validated against a full state vector at
``p = 1`` -- the agreement is a test, not an assumption.

Honest framing
--------------
This is **not** a claim of quantum advantage, and the script does not make one.
With 9 variables the optimum is found by enumerating 512 states, instantly. The
subject here is circuit construction -- how wide the circuit has to be, and what
uncleaned scratch does to the answer. A random-sampling baseline and the exact
brute-force optimum are both reported so that "QAOA did something" cannot be
mistaken for "QAOA did something useful".

Usage
-----
    python qaoa_scheduling.py
    python qaoa_scheduling.py --layers 3 --restarts 8 --seed 11
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import platform
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pennylane as qml
    from scipy.optimize import minimize
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "PennyLane and SciPy are required. Install with:\n"
        "    pip install -r requirements.txt"
    ) from exc

from uncomputation_demo import (
    ANCILLAS_PER_STEP,
    LITERALS_PER_STEP,
    Step,
    ancilla_wire_labels,
    basis_bit_table,
    compute_step,
    data_wire_labels,
    partial_trace,
    purity,
    simulate,
    von_neumann_entropy,
)

LOGGER = logging.getLogger("qaoa_scheduling")

# --------------------------------------------------------------------------
# Configuration. Documented, overridable defaults -- nothing here is tuned to
# make a result come out a particular way; every value is exposed on the CLI.
# --------------------------------------------------------------------------

DEFAULT_N_STAFF: int = 3
DEFAULT_N_SHIFTS: int = 3
DEFAULT_LAYERS: int = 1
DEFAULT_RESTARTS: int = 6
DEFAULT_SEED: int = 20240517

#: Penalty weight for violating a hard constraint, relative to the soft
#: preference costs below. Must dominate them or the optimum would be allowed to
#: buy its way out of a hard constraint.
HARD_PENALTY: float = 4.0

#: Cost of assigning one person to one shift (the soft objective: use as little
#: staff time as possible while still covering every shift).
SHIFT_COST: float = 1.0

#: Cap on the full state-vector cross-check of the naive scenario.
#: 2**22 complex128 = 64 MiB.
DEFAULT_MAX_SIM_QUBITS: int = 22

EIG_TOL: float = 1e-12

COLOR_GARBAGE: str = "#eb6834"
COLOR_CLEAN: str = "#2a78d6"
COLOR_BASELINE: str = "#7a7873"
COLOR_SURFACE: str = "#fcfcfb"
COLOR_TEXT: str = "#0b0b0b"
COLOR_TEXT_SECONDARY: str = "#52514e"
COLOR_GRID: str = "#e4e3df"


# --------------------------------------------------------------------------
# 1. The scheduling problem
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulingProblem:
    """A staff shift-scheduling instance.

    Variable ``v(staff, shift)`` is 1 when that person works that shift. Variables
    are laid out staff-major, so ``v = staff * n_shifts + shift``.

    Attributes:
        n_staff: Number of people.
        n_shifts: Number of shifts to cover.
        clauses: Three-literal hard constraints, as :class:`Step` objects whose
            conjunction evaluates to True exactly when the constraint is
            *violated*. Reusing ``Step`` means the already-validated
            ``compute_step`` subroutine builds the circuit for them unchanged.
        clause_labels: Human-readable name per clause, for reporting.
        hard_penalty: Cost charged per violated hard constraint.
        shift_cost: Cost charged per assigned (staff, shift) pair.
    """

    n_staff: int
    n_shifts: int
    clauses: Tuple[Step, ...]
    clause_labels: Tuple[str, ...]
    hard_penalty: float
    shift_cost: float

    @property
    def n_vars(self) -> int:
        """Number of binary decision variables."""
        return self.n_staff * self.n_shifts

    def variable_name(self, index: int) -> str:
        """Readable name for a decision variable."""
        return f"S{index // self.n_shifts + 1}@shift{index % self.n_shifts + 1}"


def build_problem(
    n_staff: int = DEFAULT_N_STAFF,
    n_shifts: int = DEFAULT_N_SHIFTS,
    hard_penalty: float = HARD_PENALTY,
    shift_cost: float = SHIFT_COST,
) -> SchedulingProblem:
    """Construct the scheduling instance.

    Both constraint families are expressed as three-literal conjunctions that are
    True when the constraint is broken:

    * coverage of shift ``s`` is violated when *nobody* works it, i.e.
      ``NOT x_1s AND NOT x_2s AND NOT x_3s`` -- all literals negated;
    * the no-burnout rule for person ``n`` is violated when they work *every*
      shift, i.e. ``x_n1 AND x_n2 AND x_n3`` -- no literals negated.

    Raises:
        ValueError: If the instance shape does not produce three-literal clauses.
    """
    if n_staff != LITERALS_PER_STEP or n_shifts != LITERALS_PER_STEP:
        raise ValueError(
            f"this encoding needs exactly {LITERALS_PER_STEP} staff and "
            f"{LITERALS_PER_STEP} shifts so every clause has "
            f"{LITERALS_PER_STEP} literals; got {n_staff} and {n_shifts}. "
            f"Generalising means emitting clauses of other arities, which "
            f"changes the ancilla count per clause."
        )

    clauses: List[Step] = []
    labels: List[str] = []

    # Coverage: shift s must be worked by at least one person.
    for shift in range(n_shifts):
        qubits = tuple(staff * n_shifts + shift for staff in range(n_staff))
        clauses.append(
            Step(qubits=qubits, negations=(True,) * n_staff, theta=hard_penalty)
        )
        labels.append(f"coverage(shift {shift + 1})")

    # No-burnout: person n must not work every shift.
    for staff in range(n_staff):
        qubits = tuple(staff * n_shifts + shift for shift in range(n_shifts))
        clauses.append(
            Step(qubits=qubits, negations=(False,) * n_shifts, theta=hard_penalty)
        )
        labels.append(f"no-burnout(staff {staff + 1})")

    return SchedulingProblem(
        n_staff=n_staff,
        n_shifts=n_shifts,
        clauses=tuple(clauses),
        clause_labels=tuple(labels),
        hard_penalty=hard_penalty,
        shift_cost=shift_cost,
    )


def clause_violations(problem: SchedulingProblem) -> np.ndarray:
    """Evaluate every clause on every assignment, classically.

    This is the independent reference model: it never looks at a circuit, so
    agreement between it and the simulated cost layer is real evidence.

    Returns:
        Boolean array of shape ``(n_clauses, 2 ** n_vars)``; ``True`` means the
        clause is violated by that assignment.
    """
    bits = basis_bit_table(problem.n_vars)
    out = np.empty((len(problem.clauses), 2**problem.n_vars), dtype=bool)
    for k, clause in enumerate(problem.clauses):
        acc = np.ones(2**problem.n_vars, dtype=bool)
        for qubit, negated in zip(clause.qubits, clause.negations):
            literal = bits[:, qubit].astype(bool)
            acc &= ~literal if negated else literal
        out[k] = acc
    return out


def cost_vector(problem: SchedulingProblem) -> np.ndarray:
    """Total cost ``C(x)`` for every assignment.

    ``C(x) = hard_penalty * (violated clauses) + shift_cost * (shifts assigned)``.
    """
    bits = basis_bit_table(problem.n_vars)
    violations = clause_violations(problem)
    penalty = problem.hard_penalty * violations.sum(axis=0)
    workload = problem.shift_cost * bits.sum(axis=1)
    return penalty.astype(float) + workload.astype(float)


def describe_assignment(problem: SchedulingProblem, index: int) -> str:
    """Render an assignment index as a readable roster."""
    bits = basis_bit_table(problem.n_vars)[index]
    rows = []
    for staff in range(problem.n_staff):
        shifts = [
            str(shift + 1)
            for shift in range(problem.n_shifts)
            if bits[staff * problem.n_shifts + shift]
        ]
        rows.append(f"S{staff + 1}:{','.join(shifts) if shifts else '-'}")
    return " | ".join(rows)


# --------------------------------------------------------------------------
# 2. Classical baselines and the exact optimum
# --------------------------------------------------------------------------


@dataclass
class Baselines:
    """Reference points a QAOA result must be judged against.

    Without these, "QAOA produced a distribution" reads as success when it may be
    no better than guessing.
    """

    optimum: float
    optimum_indices: Tuple[int, ...]
    random_mean: float
    worst: float
    feasible_fraction: float


def compute_baselines(problem: SchedulingProblem) -> Baselines:
    """Enumerate the whole space: exact optimum, random-guess mean, worst case.

    With ``2 ** n_vars`` assignments this is instant, which is precisely why no
    quantum advantage is claimed anywhere in this file.
    """
    costs = cost_vector(problem)
    optimum = float(costs.min())
    indices = tuple(int(i) for i in np.flatnonzero(costs == costs.min()))
    violations = clause_violations(problem)
    feasible = float(np.mean(~violations.any(axis=0)))
    return Baselines(
        optimum=optimum,
        optimum_indices=indices,
        random_mean=float(costs.mean()),
        worst=float(costs.max()),
        feasible_fraction=feasible,
    )


def normalised_score(expected_cost: float, baselines: Baselines) -> float:
    """Map an expected cost onto ``[0, 1]``: 1 = optimal, 0 = no better than random.

    ``(random_mean - E[C]) / (random_mean - optimum)``. Values can go slightly
    negative, which means *worse* than uniform guessing -- a real outcome worth
    seeing rather than clipping away.
    """
    spread = baselines.random_mean - baselines.optimum
    if spread <= 0:
        raise ValueError("degenerate instance: random mean equals the optimum")
    return (baselines.random_mean - expected_cost) / spread


# --------------------------------------------------------------------------
# 3. QAOA circuits
# --------------------------------------------------------------------------


def cost_layer_uncomputed(
    problem: SchedulingProblem,
    gamma: float,
    data_wires: Sequence[str],
    anc_t: str,
    anc_r: str,
) -> None:
    """Scenario B cost layer: compute -> phase -> ``U-dagger`` -> reuse.

    One scratch pair serves every clause. The one-local workload term needs no
    scratch and is applied directly.
    """
    for clause in problem.clauses:
        compute_step(clause, data_wires, anc_t, anc_r)
        # Violated clauses pick up the penalty phase.
        qml.PhaseShift(-gamma * problem.hard_penalty, wires=anc_r)
        # U-dagger: scratch back to |0>, ready for the next clause.
        qml.adjoint(compute_step)(clause, data_wires, anc_t, anc_r)

    for wire in data_wires:
        qml.PhaseShift(-gamma * problem.shift_cost, wires=wire)


def cost_layer_naive(
    problem: SchedulingProblem,
    gamma: float,
    data_wires: Sequence[str],
    ancillas: Sequence[str],
) -> None:
    """Scenario A cost layer: a fresh scratch pair per clause, never cleaned."""
    for k, clause in enumerate(problem.clauses):
        anc_t = ancillas[ANCILLAS_PER_STEP * k]
        anc_r = ancillas[ANCILLAS_PER_STEP * k + 1]
        compute_step(clause, data_wires, anc_t, anc_r)
        qml.PhaseShift(-gamma * problem.hard_penalty, wires=anc_r)
        # No cleanup: anc_t / anc_r stay entangled with the schedule register.

    for wire in data_wires:
        qml.PhaseShift(-gamma * problem.shift_cost, wires=wire)


def qaoa_circuit_uncomputed(
    problem: SchedulingProblem, gammas: Sequence[float], betas: Sequence[float]
) -> Callable[[], None]:
    """Full scenario-B QAOA circuit as a quantum function."""
    data = data_wire_labels(problem.n_vars)
    anc_t, anc_r = ancilla_wire_labels(ANCILLAS_PER_STEP)

    def circuit() -> None:
        for wire in data:
            qml.Hadamard(wires=wire)
        for gamma, beta in zip(gammas, betas):
            cost_layer_uncomputed(problem, gamma, data, anc_t, anc_r)
            for wire in data:
                qml.RX(2.0 * beta, wires=wire)

    return circuit


def qaoa_circuit_naive(
    problem: SchedulingProblem, gammas: Sequence[float], betas: Sequence[float]
) -> Callable[[], None]:
    """Full scenario-A QAOA circuit as a quantum function.

    Each layer allocates its own block of scratch, since none of it was cleaned.
    """
    data = data_wire_labels(problem.n_vars)
    per_layer = ANCILLAS_PER_STEP * len(problem.clauses)
    ancillas = ancilla_wire_labels(per_layer * len(gammas))

    def circuit() -> None:
        for wire in data:
            qml.Hadamard(wires=wire)
        for layer, (gamma, beta) in enumerate(zip(gammas, betas)):
            block = ancillas[layer * per_layer : (layer + 1) * per_layer]
            cost_layer_naive(problem, gamma, data, block)
            for wire in data:
                qml.RX(2.0 * beta, wires=wire)

    return circuit


def measure_widths(
    problem: SchedulingProblem, layers: int
) -> Dict[str, int]:
    """Measure both circuits' widths from the constructed tapes."""
    gammas = [0.3] * layers
    betas = [0.2] * layers

    def width(fn: Callable[[], None]) -> int:
        tape = qml.tape.make_qscript(fn)()
        return len(tape.wires)

    return {
        "naive": width(qaoa_circuit_naive(problem, gammas, betas)),
        "uncomputed": width(qaoa_circuit_uncomputed(problem, gammas, betas)),
        "n_vars": problem.n_vars,
        "n_clauses": len(problem.clauses),
    }


# --------------------------------------------------------------------------
# 4. Simulation
# --------------------------------------------------------------------------


def probabilities_uncomputed_circuit(
    problem: SchedulingProblem, gammas: Sequence[float], betas: Sequence[float]
) -> np.ndarray:
    """Outcome distribution for scenario B, from the actual PennyLane circuit.

    This is the ground truth, but it rebuilds and re-simulates an 11-qubit tape on
    every call, which is far too slow to sit inside an optimiser loop. It is used
    to *verify* the fast path below, at the optimum.
    """
    wires = data_wire_labels(problem.n_vars) + ancilla_wire_labels(ANCILLAS_PER_STEP)
    state = simulate(qaoa_circuit_uncomputed(problem, gammas, betas), wires)
    tensor = state.reshape(2**problem.n_vars, 2**ANCILLAS_PER_STEP)
    return np.sum(np.abs(tensor) ** 2, axis=1)


def probabilities_uncomputed(
    problem: SchedulingProblem, gammas: Sequence[float], betas: Sequence[float]
) -> np.ndarray:
    """Outcome distribution for scenario B, fast path.

    After uncomputation the scratch is exactly ``|00>`` and factors out, so the
    cost layer acts on the schedule register as precisely ``exp(-i gamma C(x))``
    -- which :func:`verify_cost_layer` confirms to ~1e-17 against the real
    circuit. That makes a diagonal phase plus a mixer an exact substitute, and it
    is orders of magnitude cheaper than rebuilding a tape per optimiser step.

    This shortcut is legitimate *only because* uncomputation succeeded. It has no
    analogue in the naive scenario, where the scratch does not factor out --
    which is the whole point.
    """
    dim = 2**problem.n_vars
    costs = cost_vector(problem)
    amplitudes = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
    for gamma, beta in zip(gammas, betas):
        amplitudes = np.exp(-1j * gamma * costs) * amplitudes
        amplitudes = apply_mixer_to_state(problem.n_vars, beta, amplitudes)
    return np.abs(amplitudes) ** 2


def apply_mixer_to_state(
    n_vars: int, beta: float, amplitudes: np.ndarray
) -> np.ndarray:
    """Apply ``prod_j RX(2 beta)`` qubit by qubit, without forming a dense matrix.

    Note:
        Does not modify ``amplitudes``. ``np.moveaxis`` returns a *view*, so
        writing through it would silently mutate the caller's array; each axis is
        therefore written into a fresh buffer.
    """
    single = np.array(
        [
            [np.cos(beta), -1j * np.sin(beta)],
            [-1j * np.sin(beta), np.cos(beta)],
        ],
        dtype=complex,
    )
    working = np.array(amplitudes, dtype=complex).reshape(-1)
    for axis in range(n_vars):
        _apply_axis_inplace(working, n_vars, axis, single)
    return working


def garbage_classes(problem: SchedulingProblem) -> np.ndarray:
    """Label each assignment by the garbage one cost layer would leave behind.

    The scratch pair for clause ``k`` holds ``(t_k(x), r_k(x))``. Two assignments
    keep their mutual coherence only if every one of those values agrees.

    Returns:
        Integer array of shape ``(2 ** n_vars,)``; equal labels mean identical
        garbage.
    """
    bits = basis_bit_table(problem.n_vars)
    columns = []
    for clause in problem.clauses:
        # r_k: the full conjunction (the clause violation itself)
        acc = np.ones(2**problem.n_vars, dtype=bool)
        for qubit, negated in zip(clause.qubits, clause.negations):
            literal = bits[:, qubit].astype(bool)
            acc &= ~literal if negated else literal
        # t_k: the first Toffoli's output, the partial conjunction
        partial = np.ones(2**problem.n_vars, dtype=bool)
        for qubit, negated in zip(clause.qubits[:2], clause.negations[:2]):
            literal = bits[:, qubit].astype(bool)
            partial &= ~literal if negated else literal
        columns.append(partial)
        columns.append(acc)
    stacked = np.stack(columns, axis=1)
    _, labels = np.unique(stacked, axis=0, return_inverse=True)
    return labels


def _apply_axis_inplace(
    flat: np.ndarray, n_axes: int, axis: int, matrix: np.ndarray
) -> None:
    """Apply a 2x2 matrix along one axis of a flattened rank-``n_axes`` tensor.

    Uses only ``reshape`` and slicing, so no transpose or non-contiguous copy is
    ever materialised -- ``np.moveaxis`` here costs a full buffer copy per axis,
    which dominated the runtime at depth 3. Modifies ``flat`` in place; callers
    are responsible for passing a working copy.
    """
    left = 1 << axis
    right = 1 << (n_axes - axis - 1)
    view = flat.reshape(left, 2, right)
    lower = view[:, 0, :].copy()
    upper = view[:, 1, :].copy()
    view[:, 0, :] = matrix[0, 0] * lower + matrix[0, 1] * upper
    view[:, 1, :] = matrix[1, 0] * lower + matrix[1, 1] * upper


def apply_mixer_to_density(
    n_vars: int, beta: float, rho: np.ndarray
) -> np.ndarray:
    """Return ``M rho M^dagger`` for ``M = prod_j RX(2 beta)``, axis by axis.

    Forming ``M`` densely and doing two ``2**n`` cubed matrix products costs
    ~134 Mflop per layer at ``n = 9``. Applying the single-qubit factor to each
    of the ``n`` row axes and each of the ``n`` column axes is ``O(n * 4**n)``,
    roughly fifty times cheaper, and it is what makes depth ``p >= 2`` tractable
    inside an optimiser loop.

    Equivalent to the dense form up to floating-point round-off, which
    ``test_mixer_density_matches_dense_form`` checks.
    """
    single = np.array(
        [
            [np.cos(beta), -1j * np.sin(beta)],
            [-1j * np.sin(beta), np.cos(beta)],
        ],
        dtype=complex,
    )
    working = np.array(rho, dtype=complex).reshape(-1)
    for axis in range(n_vars):
        _apply_axis_inplace(working, 2 * n_vars, axis, single)
    # (M rho M^dagger)[i, j] = sum M[i,k] rho[k,l] conj(M[j,l]) -> conjugate on
    # the column axes.
    for axis in range(n_vars, 2 * n_vars):
        _apply_axis_inplace(working, 2 * n_vars, axis, single.conj())
    return working.reshape(2**n_vars, 2**n_vars)


def mixer_matrix(n_vars: int, beta: float) -> np.ndarray:
    """``prod_j RX(2 beta)`` as a dense ``2**n_vars`` matrix."""
    single = np.array(
        [
            [np.cos(beta), -1j * np.sin(beta)],
            [-1j * np.sin(beta), np.cos(beta)],
        ],
        dtype=complex,
    )
    full = np.array([[1.0 + 0j]])
    for _ in range(n_vars):
        full = np.kron(full, single)
    return full


def probabilities_naive(
    problem: SchedulingProblem, gammas: Sequence[float], betas: Sequence[float]
) -> np.ndarray:
    """Outcome distribution for scenario A, via the exact dephasing model.

    Garbage written by a layer is a function of the data register at that moment
    and is never touched again, so it can be traced out immediately. The cost
    layer then acts on the data density matrix as a dephasing channel in the
    garbage-class basis, followed by the diagonal cost phase:

        rho -> D ( sum_c P_c rho P_c ) D^dagger

    Exact at any depth. Cross-checked against a full state vector at ``p = 1`` by
    :func:`cross_check_naive_model`.
    """
    dim = 2**problem.n_vars
    labels = garbage_classes(problem)
    costs = cost_vector(problem)

    # Start from |+>^n as a density matrix.
    amplitudes = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
    rho = np.outer(amplitudes, amplitudes.conj())

    same_class = labels[:, None] == labels[None, :]

    for gamma, beta in zip(gammas, betas):
        # Trace out this layer's garbage: coherence survives only within a class.
        rho = rho * same_class
        # Diagonal cost phase (commutes with the dephasing, applied after).
        phase = np.exp(-1j * gamma * costs)
        rho = phase[:, None] * rho * phase.conj()[None, :]
        # Mixer, applied axis by axis rather than as a dense 2**n matrix.
        rho = apply_mixer_to_density(problem.n_vars, beta, rho)

    probabilities = np.real(np.diag(rho))
    total = probabilities.sum()
    if abs(total - 1.0) > 1e-8:
        raise RuntimeError(
            f"naive model lost normalisation: total probability {total:.12f}"
        )
    return np.clip(probabilities, 0.0, None) / total


def cross_check_naive_model(
    problem: SchedulingProblem,
    gamma: float,
    beta: float,
    max_sim_qubits: int = DEFAULT_MAX_SIM_QUBITS,
) -> Optional[float]:
    """Validate the dephasing model against a full state vector at ``p = 1``.

    Returns:
        Maximum absolute probability deviation, or ``None`` if the full circuit
        exceeds the qubit budget.
    """
    per_layer = ANCILLAS_PER_STEP * len(problem.clauses)
    total_qubits = problem.n_vars + per_layer
    if total_qubits > max_sim_qubits:
        LOGGER.info(
            "naive cross-check skipped: %d qubits exceeds budget %d",
            total_qubits,
            max_sim_qubits,
        )
        return None

    wires = data_wire_labels(problem.n_vars) + ancilla_wire_labels(per_layer)
    state = simulate(qaoa_circuit_naive(problem, [gamma], [beta]), wires)
    tensor = state.reshape(2**problem.n_vars, 2**per_layer)
    exact = np.sum(np.abs(tensor) ** 2, axis=1)
    model = probabilities_naive(problem, [gamma], [beta])
    return float(np.max(np.abs(exact - model)))


def scratch_diagnostics(
    problem: SchedulingProblem, gamma: float, clause_index: int
) -> Dict[str, float]:
    """Scratch entropy and purity before and after one clause's ``U-dagger``."""
    data = data_wire_labels(problem.n_vars)
    anc_t, anc_r = ancilla_wire_labels(ANCILLAS_PER_STEP)
    wires = data + [anc_t, anc_r]
    clause = problem.clauses[clause_index]

    def make(stage: str) -> Callable[[], None]:
        def circuit() -> None:
            for wire in data:
                qml.Hadamard(wires=wire)
            for k in range(clause_index + 1):
                current = problem.clauses[k]
                compute_step(current, data, anc_t, anc_r)
                qml.PhaseShift(-gamma * problem.hard_penalty, wires=anc_r)
                if k == clause_index and stage == "compute":
                    return
                qml.adjoint(compute_step)(current, data, anc_t, anc_r)

        return circuit

    results: Dict[str, float] = {}
    for stage, label in (("compute", "before"), ("uncompute", "after")):
        state = simulate(make(stage), wires)
        rho = partial_trace(state, len(wires), [problem.n_vars, problem.n_vars + 1])
        results[f"entropy_{label}"] = von_neumann_entropy(rho)
        results[f"purity_{label}"] = purity(rho)
    _ = clause
    return results


def verify_cost_layer(problem: SchedulingProblem, gamma: float) -> float:
    """Check the uncomputed cost layer implements ``exp(-i gamma C(x))`` exactly.

    Compares the circuit's action on the uniform superposition against the
    classically computed diagonal, up to an irrelevant global phase.

    Returns:
        Maximum absolute deviation in the resulting amplitudes.
    """
    data = data_wire_labels(problem.n_vars)
    anc_t, anc_r = ancilla_wire_labels(ANCILLAS_PER_STEP)
    wires = data + [anc_t, anc_r]

    def circuit() -> None:
        for wire in data:
            qml.Hadamard(wires=wire)
        cost_layer_uncomputed(problem, gamma, data, anc_t, anc_r)

    state = simulate(circuit, wires)
    tensor = state.reshape(2**problem.n_vars, 2**ANCILLAS_PER_STEP)

    # All population must be back in the |00> scratch sector.
    leaked = float(np.sum(np.abs(tensor) ** 2) - np.sum(np.abs(tensor[:, 0]) ** 2))
    if leaked > 1e-9:
        raise RuntimeError(f"cost layer left {leaked:.3e} population in scratch")

    actual = tensor[:, 0]
    dim = 2**problem.n_vars
    expected = np.exp(-1j * gamma * cost_vector(problem)) / np.sqrt(dim)

    # Remove global phase before comparing.
    reference = actual[np.argmax(np.abs(actual))]
    target = expected[np.argmax(np.abs(actual))]
    if abs(reference) < 1e-12 or abs(target) < 1e-12:
        raise RuntimeError("cannot fix global phase: degenerate amplitude")
    actual = actual * (target / reference) / abs(target / reference)
    return float(np.max(np.abs(actual - expected)))


# --------------------------------------------------------------------------
# 5. Optimisation
# --------------------------------------------------------------------------


def expected_cost(probabilities: np.ndarray, costs: np.ndarray) -> float:
    """Expected value of the cost under an outcome distribution."""
    return float(np.dot(probabilities, costs))


@dataclass
class OptimisationResult:
    """Outcome of one QAOA parameter optimisation."""

    scenario: str
    layers: int
    best_params: Tuple[float, ...]
    expected_cost: float
    score: float
    probability_optimal: float
    per_restart_costs: Tuple[float, ...] = field(default_factory=tuple)

    @property
    def cost_spread(self) -> float:
        """Spread across restarts -- a single run is an anecdote."""
        if len(self.per_restart_costs) < 2:
            return float("nan")
        return float(np.std(self.per_restart_costs))


def optimise(
    problem: SchedulingProblem,
    scenario: str,
    layers: int,
    seed: int,
    restarts: int,
    max_iterations: int = 400,
) -> OptimisationResult:
    """Optimise ``(gamma, beta)`` by seeded multi-start COBYLA.

    Args:
        scenario: ``"uncomputed"`` or ``"naive"``.
        restarts: Independent random starting points. Every restart's result is
            retained so the spread can be reported rather than only the best.

    Raises:
        ValueError: On an unknown scenario.
    """
    if scenario not in ("uncomputed", "naive"):
        raise ValueError(f"unknown scenario {scenario!r}")

    costs = cost_vector(problem)
    baselines = compute_baselines(problem)
    distribution = (
        probabilities_uncomputed if scenario == "uncomputed" else probabilities_naive
    )

    def objective(params: np.ndarray) -> float:
        gammas = params[:layers]
        betas = params[layers:]
        return expected_cost(distribution(problem, gammas, betas), costs)

    rng = np.random.default_rng(seed)
    best_value = np.inf
    best_params: Optional[np.ndarray] = None
    per_restart: List[float] = []

    for restart in range(restarts):
        start = np.concatenate(
            [
                rng.uniform(0.0, np.pi, size=layers),
                rng.uniform(0.0, np.pi / 2.0, size=layers),
            ]
        )
        outcome = minimize(
            objective,
            start,
            method="COBYLA",
            options={"maxiter": max_iterations, "rhobeg": 0.35},
        )
        per_restart.append(float(outcome.fun))
        LOGGER.debug(
            "%s restart %d/%d: E[C] = %.6f", scenario, restart + 1, restarts,
            outcome.fun,
        )
        if outcome.fun < best_value:
            best_value = float(outcome.fun)
            best_params = np.asarray(outcome.x, dtype=float)

    if best_params is None:
        raise RuntimeError("optimisation produced no result")

    probabilities = distribution(
        problem, best_params[:layers], best_params[layers:]
    )
    optimal_mass = float(sum(probabilities[i] for i in baselines.optimum_indices))

    return OptimisationResult(
        scenario=scenario,
        layers=layers,
        best_params=tuple(float(v) for v in best_params),
        expected_cost=best_value,
        score=normalised_score(best_value, baselines),
        probability_optimal=optimal_mass,
        per_restart_costs=tuple(per_restart),
    )


# --------------------------------------------------------------------------
# 6. Reporting and plotting
# --------------------------------------------------------------------------


def print_report(
    problem: SchedulingProblem,
    baselines: Baselines,
    widths: Dict[str, int],
    results: Dict[int, Dict[str, OptimisationResult]],
    diagnostics: Dict[str, object],
    metadata: Dict[str, object],
) -> None:
    """Print every table to stdout."""
    print()
    print("=" * 78)
    print("QAOA SHIFT SCHEDULING - WITH AND WITHOUT UNCOMPUTATION")
    print("=" * 78)
    print(
        f"seed={metadata['seed']}  staff={problem.n_staff}  shifts={problem.n_shifts}"
        f"  vars={problem.n_vars}  clauses={len(problem.clauses)}"
    )
    print(
        f"pennylane={metadata['pennylane']}  numpy={metadata['numpy']}  "
        f"python={metadata['python']}"
    )
    print(
        f"hard_penalty={problem.hard_penalty}  shift_cost={problem.shift_cost}"
    )
    print()

    print("--- Problem: hard constraints (violated when the conjunction holds) ---")
    for label, clause in zip(problem.clause_labels, problem.clauses):
        literals = " AND ".join(
            ("NOT " if neg else "") + problem.variable_name(q)
            for q, neg in zip(clause.qubits, clause.negations)
        )
        print(f"  {label:<26} violated iff  {literals}")

    print()
    print("--- Classical reference (full enumeration of 2^%d assignments) ---"
          % problem.n_vars)
    print(f"  optimum cost          {baselines.optimum:.3f}")
    print(f"  optimal rosters       {len(baselines.optimum_indices)}")
    for index in baselines.optimum_indices[:4]:
        print(f"      {describe_assignment(problem, index)}")
    if len(baselines.optimum_indices) > 4:
        print(f"      ... and {len(baselines.optimum_indices) - 4} more")
    print(f"  uniform-random mean   {baselines.random_mean:.3f}")
    print(f"  worst cost            {baselines.worst:.3f}")
    print(f"  feasible fraction     {baselines.feasible_fraction:.4f}")

    print()
    print("--- Circuit width (measured from the constructed tapes) ---")
    header = f"{'layers p':>8}  {'naive':>8}  {'uncomputed':>11}  {'saved':>6}"
    print(header)
    print("-" * len(header))
    for layers in sorted(results):
        w = measure_widths(problem, layers)
        print(
            f"{layers:>8}  {w['naive']:>8}  {w['uncomputed']:>11}  "
            f"{w['naive'] - w['uncomputed']:>6}"
        )

    print()
    print("--- Cost layer correctness and scratch hygiene ---")
    print(f"  max |amplitude| deviation from exp(-i*gamma*C(x)):  "
          f"{diagnostics['cost_layer_error']:.3e}")
    print(f"  scratch entropy before U-dagger:  "
          f"{diagnostics['entropy_before']:.9f} bits")
    print(f"  scratch entropy after  U-dagger:  "
          f"{diagnostics['entropy_after']:.3e} bits")
    print(f"  scratch purity  after  U-dagger:  "
          f"{diagnostics['purity_after']:.9f}")
    deviation = diagnostics.get("naive_model_deviation")
    if deviation is None:
        print("  naive dephasing model vs full state vector:  not run (over budget)")
    else:
        print(f"  naive dephasing model vs full state vector:  {deviation:.3e}")
    print(f"  optimiser fast path vs real PennyLane circuit:   "
          f"{diagnostics['fast_path_deviation']:.3e}")

    print()
    print("--- Solution quality ---")
    print("    score: 1 = optimum, 0 = no better than uniform random guessing")
    header = (
        f"{'p':>3}  {'scenario':>11}  {'E[cost]':>9}  {'score':>7}  "
        f"{'P(optimal)':>10}  {'spread':>7}"
    )
    print(header)
    print("-" * len(header))
    for layers in sorted(results):
        for scenario in ("uncomputed", "naive"):
            r = results[layers][scenario]
            print(
                f"{layers:>3}  {scenario:>11}  {r.expected_cost:>9.4f}  "
                f"{r.score:>7.4f}  {r.probability_optimal:>10.4f}  "
                f"{r.cost_spread:>7.4f}"
            )
        print(
            f"{'':>3}  {'random':>11}  {baselines.random_mean:>9.4f}  "
            f"{0.0:>7.4f}  "
            f"{len(baselines.optimum_indices) / 2 ** problem.n_vars:>10.4f}  "
            f"{'-':>7}"
        )
    print()


def make_plot(
    problem: SchedulingProblem,
    baselines: Baselines,
    results: Dict[int, Dict[str, OptimisationResult]],
    metadata: Dict[str, object],
    output_path: str,
) -> None:
    """Write the two-panel comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    layer_values = sorted(results)
    naive_widths = [measure_widths(problem, p)["naive"] for p in layer_values]
    unc_widths = [measure_widths(problem, p)["uncomputed"] for p in layer_values]
    naive_scores = [results[p]["naive"].score for p in layer_values]
    unc_scores = [results[p]["uncomputed"].score for p in layer_values]

    figure, (ax_width, ax_score) = plt.subplots(
        1, 2, figsize=(12.0, 4.8), dpi=200, facecolor=COLOR_SURFACE
    )

    for axis in (ax_width, ax_score):
        axis.set_facecolor(COLOR_SURFACE)
        axis.grid(True, color=COLOR_GRID, linewidth=0.8, linestyle="-")
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(COLOR_GRID)
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=9, length=0)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    marker = dict(marker="o", markersize=6, markeredgecolor=COLOR_SURFACE,
                  markeredgewidth=2, linewidth=2)

    ax_width.plot(layer_values, naive_widths, color=COLOR_GARBAGE,
                  label="Without uncomputation", **marker)
    ax_width.plot(layer_values, unc_widths, color=COLOR_CLEAN,
                  label="With uncomputation", **marker)
    for x, y in ((layer_values[-1], naive_widths[-1]),
                 (layer_values[-1], unc_widths[-1])):
        ax_width.annotate(f"{y} qubits", xy=(x, y), xytext=(-4, 9),
                          textcoords="offset points", fontsize=9,
                          color=COLOR_TEXT, ha="right")
    ax_width.set_xlabel("QAOA layers p", fontsize=10, color=COLOR_TEXT)
    ax_width.set_ylabel("Physical qubits required", fontsize=10, color=COLOR_TEXT)
    ax_width.set_title(
        f"(a) Circuit width  ({problem.n_vars} decision variables, "
        f"{len(problem.clauses)} 3-literal clauses)",
        fontsize=11, color=COLOR_TEXT, loc="left", pad=10,
    )
    ax_width.set_ylim(0, max(naive_widths) * 1.18)
    ax_width.legend(frameon=False, fontsize=9, loc="upper left",
                    labelcolor=COLOR_TEXT_SECONDARY)

    ax_score.axhline(0.0, color=COLOR_BASELINE, linewidth=1.4)
    ax_score.annotate("uniform random guessing", xy=(layer_values[0], 0.0),
                      xytext=(0, 7), textcoords="offset points", fontsize=8.5,
                      color=COLOR_BASELINE)
    ax_score.plot(layer_values, naive_scores, color=COLOR_GARBAGE,
                  label="Without uncomputation", **marker)
    ax_score.plot(layer_values, unc_scores, color=COLOR_CLEAN,
                  label="With uncomputation", **marker)
    ax_score.set_xlabel("QAOA layers p", fontsize=10, color=COLOR_TEXT)
    ax_score.set_ylabel("Score  (1 = optimum, 0 = random)", fontsize=10,
                        color=COLOR_TEXT)
    ax_score.set_title("(b) Solution quality", fontsize=11, color=COLOR_TEXT,
                       loc="left", pad=10)
    top = max(max(unc_scores), max(naive_scores), 0.1)
    bottom = min(min(unc_scores), min(naive_scores), 0.0)
    ax_score.set_ylim(bottom - 0.08, top * 1.25)
    ax_score.legend(frameon=False, fontsize=9, loc="upper left",
                    labelcolor=COLOR_TEXT_SECONDARY)

    figure.text(
        0.01, 0.015,
        f"seed={metadata['seed']} | PennyLane {metadata['pennylane']}, "
        f"NumPy {metadata['numpy']}, Python {metadata['python']} | "
        f"exact optimum {baselines.optimum:.2f} by full enumeration - "
        f"no quantum advantage is claimed",
        fontsize=7, color=COLOR_TEXT_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_path, facecolor=COLOR_SURFACE)
    plt.close(figure)
    LOGGER.info("wrote %s", output_path)


# --------------------------------------------------------------------------
# 7. CLI
# --------------------------------------------------------------------------


def run(
    layers: int,
    seed: int,
    restarts: int,
    max_sim_qubits: int,
    max_iterations: int = 400,
) -> Tuple[SchedulingProblem, Baselines, Dict[int, Dict[str, OptimisationResult]],
           Dict[str, object], Dict[str, object]]:
    """Run the whole study."""
    problem = build_problem()
    baselines = compute_baselines(problem)

    diagnostics: Dict[str, object] = {}
    diagnostics["cost_layer_error"] = verify_cost_layer(problem, gamma=0.37)
    scratch = scratch_diagnostics(problem, gamma=0.37, clause_index=0)
    diagnostics.update(scratch)
    diagnostics["naive_model_deviation"] = cross_check_naive_model(
        problem, gamma=0.37, beta=0.21, max_sim_qubits=max_sim_qubits
    )
    # The fast path used inside the optimiser must match the real circuit.
    diagnostics["fast_path_deviation"] = float(
        np.max(
            np.abs(
                probabilities_uncomputed(problem, [0.37], [0.21])
                - probabilities_uncomputed_circuit(problem, [0.37], [0.21])
            )
        )
    )

    results: Dict[int, Dict[str, OptimisationResult]] = {}
    for depth in range(1, layers + 1):
        results[depth] = {}
        for scenario in ("uncomputed", "naive"):
            LOGGER.info("optimising p=%d, %s", depth, scenario)
            results[depth][scenario] = optimise(
                problem, scenario, depth, seed=seed + depth,
                restarts=restarts, max_iterations=max_iterations,
            )

    metadata: Dict[str, object] = {
        "seed": seed,
        "layers": layers,
        "restarts": restarts,
        "optimiser_maxiter": max_iterations,
        "max_sim_qubits": max_sim_qubits,
        "hard_penalty": problem.hard_penalty,
        "shift_cost": problem.shift_cost,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pennylane": qml.version(),
        "note": (
            "No quantum advantage is claimed. The optimum is found by "
            "enumerating 2**n_vars assignments classically."
        ),
    }
    return problem, baselines, results, diagnostics, metadata


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "QAOA for staff shift scheduling, comparing a naive cost layer "
            "against one that uncomputes its scratch with U-dagger."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS,
                        help="maximum QAOA depth p to study")
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS,
                        help="optimiser restarts per configuration")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-sim-qubits", type=int,
                        default=DEFAULT_MAX_SIM_QUBITS)
    parser.add_argument("--maxiter", type=int, default=400,
                        help="COBYLA iteration cap per restart")
    parser.add_argument("--plot", default="qaoa_scheduling_comparison.png")
    parser.add_argument("--json", default="qaoa_results.json")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        problem, baselines, results, diagnostics, metadata = run(
            layers=args.layers,
            seed=args.seed,
            restarts=args.restarts,
            max_sim_qubits=args.max_sim_qubits,
            max_iterations=args.maxiter,
        )
    except (ValueError, RuntimeError) as exc:
        LOGGER.error("run failed: %s", exc)
        return 1

    widths = measure_widths(problem, args.layers)
    print_report(problem, baselines, widths, results, diagnostics, metadata)

    payload = {
        "metadata": metadata,
        "baselines": {
            "optimum": baselines.optimum,
            "optimum_indices": list(baselines.optimum_indices),
            "random_mean": baselines.random_mean,
            "worst": baselines.worst,
            "feasible_fraction": baselines.feasible_fraction,
        },
        "diagnostics": diagnostics,
        "widths": {str(p): measure_widths(problem, p) for p in sorted(results)},
        "results": {
            str(p): {
                scenario: {
                    "expected_cost": r.expected_cost,
                    "score": r.score,
                    "probability_optimal": r.probability_optimal,
                    "best_params": list(r.best_params),
                    "per_restart_costs": list(r.per_restart_costs),
                    "cost_spread": r.cost_spread,
                }
                for scenario, r in by_scenario.items()
            }
            for p, by_scenario in results.items()
        },
    }
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    LOGGER.info("wrote %s", args.json)

    if not args.no_plot:
        try:
            make_plot(problem, baselines, results, metadata, args.plot)
        except ImportError:
            LOGGER.error("matplotlib is required for --plot")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
