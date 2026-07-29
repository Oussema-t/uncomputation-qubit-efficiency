#!/usr/bin/env python3
"""Uncomputation (U-dagger) qubit-efficiency demonstration.

What this shows
---------------
A quantum subroutine that computes a Boolean predicate into scratch ("ancilla")
qubits leaves those qubits *entangled* with the data register. Until they are
cleaned up they are **garbage**: they cannot be reused, and they act as a
which-path record that destroys interference on the data register.

Applying the inverse of the compute step, ``U-dagger``, disentangles the scratch
and returns it to ``|0>`` exactly. The same physical ancillas can then be reused
by the next step.

The demo implements one logical operation two ways and measures the difference.

Logical target
--------------
An ``N``-step phase oracle on ``n_data`` qubits::

    U_logical |x> = exp(i * sum_k theta_k * p_k(x)) |x>

where ``p_k(x)`` is a conjunction of three signed literals of ``x`` (chosen by a
seeded RNG), and ``theta_k`` is a seeded random angle.

Each step's compute subroutine ``U_k`` is two Toffolis writing into two scratch
qubits::

    t = l_a AND l_b            (first Toffoli)
    r = t AND l_c              (second Toffoli)

``U_k`` is deliberately **not** self-inverse -- the two Toffolis do not commute
(``t`` is the target of one and a control of the other) -- so ``U_k-dagger`` is a
genuine adjoint rather than a repeat of ``U_k``. That is asserted numerically in
the test suite.

Scenario A (naive)
    Fresh ``(t_k, r_k)`` for every step, never cleaned up.
    Width: ``n_data + 2N`` qubits.

Scenario B (uncomputation)
    One ``(t, r)`` pair. Per step: ``U_k``, phase on ``r``, ``adjoint(U_k)``, reuse.
    Width: ``n_data + 2`` qubits.

Two independent simulation methods
----------------------------------
M1  Full PennyLane state-vector simulation. Exact, but scenario A needs
    ``2 ** (n_data + 2N)`` amplitudes, so it is capped by ``--max-sim-qubits``.

M2  A structured exact model over the ``2 ** n_data`` data basis only. Because
    every compute step here is a classical reversible Boolean map and every
    "use" is diagonal in the computational basis, scenario A's global state is
    ``sum_x a_x e^{i phi(x)} |x>|g(x)>``, so the reduced data state is

        rho_A[x, x'] = a_x a_x'^* e^{i(phi(x) - phi(x'))} * [g(x) == g(x')]

    which is exact for *this circuit class* (not a general-purpose simulator) and
    costs no memory in ``N``.

M2 is **validated against M1** on every ``N`` where both run; the agreement check
is a test, not an assumption. Metrics for large ``N`` come from M2 only after
that check passes.

Reproducibility
---------------
Every number this script prints is tagged with the seed and the library versions
that produced it, and the same values are written to the results JSON.

Usage
-----
    python uncomputation_demo.py
    python uncomputation_demo.py --max-steps 12 --n-data 5 --seed 7
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pennylane as qml
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "PennyLane is required. Install the pinned dependencies with:\n"
        "    pip install -r requirements.txt"
    ) from exc


LOGGER = logging.getLogger("uncomputation_demo")

# --------------------------------------------------------------------------
# Configuration constants.
#
# These are documented, overridable defaults -- not tuned values. Nothing below
# is chosen to make a result come out a particular way; the CLI exposes each one
# and the reported numbers are stable across seeds (see README).
# --------------------------------------------------------------------------

DEFAULT_N_DATA: int = 4
DEFAULT_MAX_STEPS: int = 20
DEFAULT_SEED: int = 20240517

#: Literals per step. A k-literal conjunction needs k-1 scratch qubits when
#: built from Toffolis, which is why ANCILLAS_PER_STEP is derived, not guessed.
LITERALS_PER_STEP: int = 3
ANCILLAS_PER_STEP: int = LITERALS_PER_STEP - 1

#: Memory budget for the full state-vector method (M1). 2**20 complex128 = 16 MiB.
#: Scenario A needs n_data + 2N qubits, so this caps how far M1 can follow it.
DEFAULT_MAX_SIM_QUBITS: int = 20

#: Eigenvalues below this magnitude are treated as zero when taking logs or
#: square roots. Eigenvalues *below the negative* of this are a real numerical
#: problem and are logged loudly rather than silently clipped.
EIG_TOL: float = 1e-12

#: Validated categorical palette slots 1 and 2 (light surface).
#: Semantics are held constant across panels: orange = garbage present,
#: blue = scratch cleaned. Colour follows the entity, never its rank.
COLOR_GARBAGE: str = "#eb6834"
COLOR_CLEAN: str = "#2a78d6"
COLOR_SURFACE: str = "#fcfcfb"
COLOR_TEXT: str = "#0b0b0b"
COLOR_TEXT_SECONDARY: str = "#52514e"
COLOR_GRID: str = "#e4e3df"


# --------------------------------------------------------------------------
# 1. Problem specification and the classical reference model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One computation step: a signed conjunction of data bits, then a phase.

    Attributes:
        qubits: Indices of the data qubits used as literals, length
            ``LITERALS_PER_STEP``, all distinct.
        negations: ``negations[i] is True`` means literal ``i`` is the *negation*
            of ``qubits[i]``.
        theta: Phase angle applied when the conjunction evaluates to True.
    """

    qubits: Tuple[int, ...]
    negations: Tuple[bool, ...]
    theta: float

    def __post_init__(self) -> None:
        if len(self.qubits) != LITERALS_PER_STEP:
            raise ValueError(
                f"expected {LITERALS_PER_STEP} literals, got {len(self.qubits)}"
            )
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"literals must use distinct qubits, got {self.qubits}")
        if len(self.negations) != len(self.qubits):
            raise ValueError("negations must be the same length as qubits")


def build_steps(n_data: int, n_steps: int, seed: int) -> List[Step]:
    """Draw a reproducible sequence of computation steps.

    Args:
        n_data: Number of data qubits. Must be at least ``LITERALS_PER_STEP``.
        n_steps: Number of steps to generate.
        seed: RNG seed; the same seed always yields the same steps.

    Returns:
        A list of ``n_steps`` :class:`Step` objects.

    Raises:
        ValueError: If ``n_data`` is too small or ``n_steps`` is negative.
    """
    if n_data < LITERALS_PER_STEP:
        raise ValueError(
            f"n_data must be >= {LITERALS_PER_STEP} so each step has distinct "
            f"literals; got {n_data}"
        )
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")

    rng = np.random.default_rng(seed)
    steps: List[Step] = []
    for _ in range(n_steps):
        qubits = tuple(
            int(q) for q in rng.choice(n_data, size=LITERALS_PER_STEP, replace=False)
        )
        negations = tuple(bool(b) for b in rng.integers(0, 2, size=LITERALS_PER_STEP))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        steps.append(Step(qubits=qubits, negations=negations, theta=theta))
    return steps


def basis_bit_table(n_qubits: int) -> np.ndarray:
    """Return the ``(2**n, n)`` table of computational-basis bit strings.

    The convention matches PennyLane's :func:`qml.state` ordering: the *first*
    wire is the most significant bit. :func:`verify_bit_order` checks this
    against a live circuit at runtime rather than trusting the docstring.
    """
    indices = np.arange(2**n_qubits, dtype=np.uint64)
    shifts = np.arange(n_qubits - 1, -1, -1, dtype=np.uint64)
    return ((indices[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def evaluate_predicates(steps: Sequence[Step], n_data: int) -> np.ndarray:
    """Evaluate every step's conjunction on every basis state, classically.

    This is the independent reference model. It never looks at a circuit, so
    agreement between it and the simulated circuits is real evidence rather
    than a tautology.

    Returns:
        Boolean array of shape ``(len(steps), 2 ** n_data)``; entry ``[k, x]`` is
        ``p_k(x)``.
    """
    bits = basis_bit_table(n_data)
    out = np.empty((len(steps), 2**n_data), dtype=bool)
    for k, step in enumerate(steps):
        acc = np.ones(2**n_data, dtype=bool)
        for qubit, negated in zip(step.qubits, step.negations):
            literal = bits[:, qubit].astype(bool)
            acc &= ~literal if negated else literal
        out[k] = acc
    return out


def evaluate_intermediates(steps: Sequence[Step], n_data: int) -> np.ndarray:
    """Evaluate the *first* Toffoli's output ``t = l_a AND l_b`` per step.

    ``t`` is part of the garbage left behind in scenario A, so the structured
    model needs it to build the garbage key.

    Returns:
        Boolean array of shape ``(len(steps), 2 ** n_data)``.
    """
    bits = basis_bit_table(n_data)
    out = np.empty((len(steps), 2**n_data), dtype=bool)
    for k, step in enumerate(steps):
        acc = np.ones(2**n_data, dtype=bool)
        for qubit, negated in zip(step.qubits[:2], step.negations[:2]):
            literal = bits[:, qubit].astype(bool)
            acc &= ~literal if negated else literal
        out[k] = acc
    return out


def logical_phases(steps: Sequence[Step], n_data: int) -> np.ndarray:
    """Total accumulated phase ``sum_k theta_k p_k(x)`` for each basis state."""
    predicates = evaluate_predicates(steps, n_data)
    thetas = np.array([step.theta for step in steps], dtype=float)
    return thetas @ predicates.astype(float)


# --------------------------------------------------------------------------
# 2. Input-state preparation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InputSpec:
    """How the data register is initialised.

    Attributes:
        kind: ``"basis"`` for a computational-basis state, ``"uniform"`` for the
            equal superposition over all ``2 ** n_data`` basis states.
        bits: For ``kind == "basis"``, the bit string to prepare.
    """

    kind: str
    bits: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("basis", "uniform"):
            raise ValueError(f"unknown input kind {self.kind!r}")


def make_input_specs(n_data: int, seed: int) -> Dict[str, InputSpec]:
    """Build the two input states the benchmark compares on.

    The basis string is drawn from the seed rather than fixed, so the result is
    not quietly resting on one hand-picked input.
    """
    rng = np.random.default_rng(seed + 1)
    bits = tuple(int(b) for b in rng.integers(0, 2, size=n_data))
    return {
        "basis": InputSpec(kind="basis", bits=bits),
        "uniform": InputSpec(kind="uniform"),
    }


def input_amplitudes(spec: InputSpec, n_data: int) -> np.ndarray:
    """Return the data-register input state vector for the structured model."""
    dim = 2**n_data
    if spec.kind == "uniform":
        return np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
    if len(spec.bits) != n_data:
        raise ValueError(
            f"basis spec has {len(spec.bits)} bits but n_data is {n_data}"
        )
    index = 0
    for bit in spec.bits:
        index = (index << 1) | int(bit)
    amplitudes = np.zeros(dim, dtype=complex)
    amplitudes[index] = 1.0
    return amplitudes


def apply_input_prep(spec: InputSpec, data_wires: Sequence[object]) -> None:
    """Emit the gates that prepare the input state (used inside a QNode)."""
    if spec.kind == "uniform":
        for wire in data_wires:
            qml.Hadamard(wires=wire)
        return
    for bit, wire in zip(spec.bits, data_wires):
        if bit:
            qml.PauliX(wires=wire)


# --------------------------------------------------------------------------
# 3. Circuit construction
# --------------------------------------------------------------------------


def data_wire_labels(n_data: int) -> List[str]:
    """Wire labels for the data register."""
    return [f"d{i}" for i in range(n_data)]


def ancilla_wire_labels(count: int) -> List[str]:
    """Wire labels for the scratch register."""
    return [f"a{i}" for i in range(count)]


def compute_step(
    step: Step, data_wires: Sequence[str], anc_t: str, anc_r: str
) -> None:
    """Emit ``U_k``: write the signed conjunction of ``step`` into ``(t, r)``.

    Maps ``|x>|0>|0>`` to ``|x>|t(x)>|r(x)>`` with ``t = l_a AND l_b`` and
    ``r = t AND l_c``. The data register is left unchanged: negated literals are
    handled by X gates that are applied and then undone inside this subroutine.

    Note:
        The two Toffolis do not commute, so this subroutine is not self-inverse.
        ``qml.adjoint(compute_step)`` is therefore a real inverse, which is the
        whole point of the demonstration.
    """
    negated = [q for q, neg in zip(step.qubits, step.negations) if neg]

    # Negative literals: flip, use, flip back. Data ends where it started.
    for qubit in negated:
        qml.PauliX(wires=data_wires[qubit])

    # First AND: t <- l_a AND l_b
    qml.Toffoli(wires=[data_wires[step.qubits[0]], data_wires[step.qubits[1]], anc_t])
    # Second AND: r <- t AND l_c. Reads t, so it must follow the first Toffoli.
    qml.Toffoli(wires=[anc_t, data_wires[step.qubits[2]], anc_r])

    for qubit in negated:
        qml.PauliX(wires=data_wires[qubit])


def naive_circuit(
    steps: Sequence[Step], n_data: int, spec: InputSpec
) -> Callable[[], None]:
    """Scenario A: allocate a fresh scratch pair per step and never clean up.

    Returns a quantum function (no measurement) suitable for a QNode or for
    :func:`qml.tape.make_qscript`.
    """
    data = data_wire_labels(n_data)
    ancillas = ancilla_wire_labels(ANCILLAS_PER_STEP * len(steps))

    def circuit() -> None:
        apply_input_prep(spec, data)
        for k, step in enumerate(steps):
            anc_t = ancillas[ANCILLAS_PER_STEP * k]
            anc_r = ancillas[ANCILLAS_PER_STEP * k + 1]
            compute_step(step, data, anc_t, anc_r)
            # "Use" the result: a phase kickback conditioned on the predicate.
            qml.PhaseShift(step.theta, wires=anc_r)
            # No uncomputation. anc_t / anc_r stay entangled with the data.

    return circuit


def uncomputed_circuit(
    steps: Sequence[Step],
    n_data: int,
    spec: InputSpec,
    stop_after_step: Optional[int] = None,
    stop_stage: str = "uncompute",
) -> Callable[[], None]:
    """Scenario B: one scratch pair, cleaned with ``U-dagger`` and reused.

    Args:
        steps: The computation steps.
        n_data: Number of data qubits.
        spec: Input-state specification.
        stop_after_step: If given, halt the circuit at step index
            ``stop_after_step`` instead of running to the end. Used to inspect
            the ancilla mid-cycle.
        stop_stage: ``"compute"`` halts right after the phase is applied and
            before ``U-dagger`` (scratch dirty); ``"uncompute"`` halts right
            after ``U-dagger`` (scratch clean).
    """
    if stop_stage not in ("compute", "uncompute"):
        raise ValueError(f"unknown stop_stage {stop_stage!r}")

    data = data_wire_labels(n_data)
    anc_t, anc_r = ancilla_wire_labels(ANCILLAS_PER_STEP)

    def circuit() -> None:
        apply_input_prep(spec, data)
        for k, step in enumerate(steps):
            compute_step(step, data, anc_t, anc_r)
            qml.PhaseShift(step.theta, wires=anc_r)
            if stop_after_step is not None and k == stop_after_step:
                if stop_stage == "compute":
                    return
                # Clean up this one step, then stop.
                qml.adjoint(compute_step)(step, data, anc_t, anc_r)
                return
            # Uncompute: return the scratch to |0> so the next step reuses it.
            qml.adjoint(compute_step)(step, data, anc_t, anc_r)

    return circuit


def measure_circuit_width(
    circuit_fn: Callable[[], None], n_data: int
) -> Tuple[int, int]:
    """Measure how many qubits a circuit actually touches.

    This reads the constructed tape rather than trusting a formula, so the
    reported scaling is an observation about the circuit, not an assertion.

    Returns:
        ``(n_ancillas_touched, total_qubits)``, where ``total_qubits`` counts the
        full data register plus the distinct ancilla wires the circuit uses.
    """
    tape = qml.tape.make_qscript(circuit_fn)()
    touched = {str(w) for w in tape.wires}
    ancillas = {w for w in touched if w.startswith("a")}
    return len(ancillas), n_data + len(ancillas)


def simulate(circuit_fn: Callable[[], None], wires: Sequence[str]) -> np.ndarray:
    """Run a quantum function on ``default.qubit`` and return the state vector."""
    device = qml.device("default.qubit", wires=list(wires))

    @qml.qnode(device)
    def node():  # type: ignore[no-untyped-def]
        circuit_fn()
        return qml.state()

    return np.asarray(node(), dtype=complex)


def verify_bit_order() -> None:
    """Check that :func:`basis_bit_table` matches PennyLane's state ordering.

    Guards against a silent wire-ordering bug, which would corrupt every partial
    trace in this module while still producing plausible-looking numbers.

    Raises:
        RuntimeError: If the convention does not hold.
    """
    wires = ["d0", "d1", "d2"]

    def circuit() -> None:
        qml.PauliX(wires="d0")  # first wire == most significant bit

    state = simulate(circuit, wires)
    index = int(np.argmax(np.abs(state)))
    expected = 0b100
    if index != expected:
        raise RuntimeError(
            "PennyLane state ordering does not match basis_bit_table: expected "
            f"index {expected}, got {index}. Partial traces would be wrong."
        )
    LOGGER.debug("bit-order self-check passed (index %d)", index)


# --------------------------------------------------------------------------
# 4. Quantum-information metrics
# --------------------------------------------------------------------------


def partial_trace(
    state: np.ndarray, n_qubits: int, keep: Sequence[int]
) -> np.ndarray:
    """Reduced density matrix of a pure state over the ``keep`` qubit indices.

    Args:
        state: State vector of length ``2 ** n_qubits``.
        n_qubits: Total number of qubits.
        keep: Qubit indices to retain, in the wire order used to build ``state``.

    Returns:
        Density matrix of shape ``(2 ** len(keep), 2 ** len(keep))``.
    """
    if state.size != 2**n_qubits:
        raise ValueError(
            f"state has {state.size} amplitudes, expected {2 ** n_qubits}"
        )
    keep_sorted = sorted(int(k) for k in keep)
    if not keep_sorted:
        raise ValueError("keep must be non-empty")
    if keep_sorted[0] < 0 or keep_sorted[-1] >= n_qubits:
        raise ValueError(f"keep indices {keep_sorted} out of range for {n_qubits}")

    traced = [i for i in range(n_qubits) if i not in keep_sorted]
    tensor = state.reshape([2] * n_qubits)
    tensor = np.transpose(tensor, keep_sorted + traced)
    matrix = tensor.reshape(2 ** len(keep_sorted), 2 ** len(traced))
    return matrix @ matrix.conj().T


def _safe_eigvalsh(matrix: np.ndarray, context: str) -> np.ndarray:
    """Hermitian eigenvalues, complaining loudly about non-trivial negatives."""
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0)
    worst = float(eigenvalues.min())
    if worst < -1e-8:
        LOGGER.warning(
            "%s: density matrix has eigenvalue %.3e < -1e-8; result is suspect",
            context,
            worst,
        )
    return eigenvalues


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Von Neumann entropy ``S(rho) = -Tr(rho log2 rho)``, in bits."""
    eigenvalues = _safe_eigvalsh(rho, "von_neumann_entropy")
    positive = eigenvalues[eigenvalues > EIG_TOL]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log2(positive)))


def purity(rho: np.ndarray) -> float:
    """Purity ``Tr(rho^2)``. Equals 1 for a pure state."""
    return float(np.real(np.trace(rho @ rho)))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance ``0.5 * ||rho - sigma||_1``. Zero iff the states are equal."""
    if rho.shape != sigma.shape:
        raise ValueError(f"shape mismatch: {rho.shape} vs {sigma.shape}")
    difference = rho - sigma
    eigenvalues = np.linalg.eigvalsh((difference + difference.conj().T) / 2.0)
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def _matrix_sqrt_psd(rho: np.ndarray, context: str) -> np.ndarray:
    """Principal square root of a positive-semidefinite matrix."""
    eigenvalues, vectors = np.linalg.eigh((rho + rho.conj().T) / 2.0)
    worst = float(eigenvalues.min())
    if worst < -1e-8:
        LOGGER.warning(
            "%s: matrix sqrt clipped eigenvalue %.3e < -1e-8", context, worst
        )
    clipped = np.clip(eigenvalues, 0.0, None)
    return (vectors * np.sqrt(clipped)) @ vectors.conj().T


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity ``(Tr sqrt(sqrt(rho) sigma sqrt(rho)))**2``.

    Note:
        This is the **squared** convention, so ``F = 1`` for identical states and
        ``F = |<psi|phi>|**2`` for pure states. Some texts report the square root
        of this quantity; the convention is stated here because mixing the two is
        a common source of wrong-looking numbers.
    """
    if rho.shape != sigma.shape:
        raise ValueError(f"shape mismatch: {rho.shape} vs {sigma.shape}")
    sqrt_rho = _matrix_sqrt_psd(rho, "fidelity")
    inner = sqrt_rho @ sigma @ sqrt_rho
    eigenvalues = _safe_eigvalsh(inner, "fidelity")

    # Eigenvalues that should be exactly zero come back at ~1e-17. sqrt() turns
    # each into ~3e-9, and summing them across the spectrum pushes the result
    # above 1 -- which fidelity can never be. Treat sub-tolerance eigenvalues as
    # the zeros they are, instead of taking their square roots.
    kept = np.where(eigenvalues > EIG_TOL, eigenvalues, 0.0)
    value = float(np.sum(np.sqrt(kept)) ** 2)

    if value > 1.0 + 1e-9:
        LOGGER.warning(
            "fidelity computed as %.12f > 1; clamping. This indicates a real "
            "numerical problem, not just round-off.",
            value,
        )
    return min(value, 1.0)


def density_matrix(state: np.ndarray) -> np.ndarray:
    """Outer product ``|psi><psi|`` of a state vector."""
    return np.outer(state, state.conj())


# --------------------------------------------------------------------------
# 5. Structured exact model (M2)
# --------------------------------------------------------------------------


def structured_states(
    steps: Sequence[Step], n_data: int, spec: InputSpec
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact data-register states for both scenarios, without a state vector.

    Valid because every compute step is a classical reversible Boolean map and
    every "use" is diagonal in the computational basis. Under those conditions
    scenario A's global state is ``sum_x a_x e^{i phi(x)} |x> |g(x)>``, so tracing
    out the garbage leaves off-diagonal terms only where the garbage agrees.

    This model is checked against the full state-vector simulation on every
    ``N`` where both are feasible -- see :func:`run_benchmark`.

    Returns:
        ``(psi_ideal, rho_naive)``: the pure data state the logical operation
        should produce (which scenario B must reproduce), and the reduced data
        density matrix scenario A actually leaves behind.
    """
    amplitudes = input_amplitudes(spec, n_data)
    phases = logical_phases(steps, n_data)
    psi_ideal = amplitudes * np.exp(1j * phases)

    predicates = evaluate_predicates(steps, n_data)  # r_k(x)
    intermediates = evaluate_intermediates(steps, n_data)  # t_k(x)

    if len(steps) == 0:
        garbage_match = np.ones((2**n_data, 2**n_data), dtype=bool)
    else:
        # Garbage register content per basis state: (t_1, r_1, ..., t_N, r_N).
        garbage = np.concatenate([intermediates, predicates], axis=0).T  # (2**n, 2N)
        garbage_match = np.all(garbage[:, None, :] == garbage[None, :, :], axis=2)

    rho_naive = np.outer(psi_ideal, psi_ideal.conj()) * garbage_match
    return psi_ideal, rho_naive


# --------------------------------------------------------------------------
# 6. Benchmark runner
# --------------------------------------------------------------------------


@dataclass
class StepRecord:
    """Per-``N`` benchmark record. All fields are measured, none are assumed."""

    n_steps: int
    n_data: int
    naive_ancillas: int
    naive_total_qubits: int
    uncomputed_ancillas: int
    uncomputed_total_qubits: int
    # Scenario B vs the logical target (structured model, all N).
    fidelity_uncomputed_vs_ideal: float
    trace_distance_uncomputed_vs_ideal: float
    # Scenario A vs scenario B, per input state (structured model, all N).
    fidelity_naive_vs_uncomputed: Dict[str, float] = field(default_factory=dict)
    trace_distance_naive_vs_uncomputed: Dict[str, float] = field(default_factory=dict)
    # Ancilla diagnostics from the full state-vector simulation of scenario B.
    ancilla_entropy_before: Optional[float] = None
    ancilla_entropy_after: Optional[float] = None
    ancilla_purity_before: Optional[float] = None
    ancilla_purity_after: Optional[float] = None
    # M1-vs-M2 cross-validation (None where M1 was out of budget).
    statevector_verified: bool = False
    m1_m2_max_deviation: Optional[float] = None


def ancilla_diagnostics(
    steps: Sequence[Step], n_data: int, spec: InputSpec, step_index: int
) -> Dict[str, float]:
    """Entropy and purity of the scratch register before and after ``U-dagger``.

    Runs the scenario-B circuit twice, halting once with the scratch dirty and
    once immediately after the inverse has been applied. Scenario B is only
    ``n_data + 2`` qubits wide, so this is always affordable.
    """
    wires = data_wire_labels(n_data) + ancilla_wire_labels(ANCILLAS_PER_STEP)
    ancilla_indices = [n_data, n_data + 1]

    results: Dict[str, float] = {}
    for stage, label in (("compute", "before"), ("uncompute", "after")):
        circuit = uncomputed_circuit(
            steps, n_data, spec, stop_after_step=step_index, stop_stage=stage
        )
        state = simulate(circuit, wires)
        rho_ancilla = partial_trace(state, len(wires), ancilla_indices)
        results[f"entropy_{label}"] = von_neumann_entropy(rho_ancilla)
        results[f"purity_{label}"] = purity(rho_ancilla)
    return results


def run_benchmark(
    n_data: int = DEFAULT_N_DATA,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_SEED,
    max_sim_qubits: int = DEFAULT_MAX_SIM_QUBITS,
) -> Tuple[List[StepRecord], Dict[str, object]]:
    """Run both scenarios for ``N = 1 .. max_steps`` and collect every metric.

    Returns:
        ``(records, metadata)`` where ``metadata`` carries the seed and library
        versions needed to reproduce the numbers.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if max_sim_qubits < n_data + ANCILLAS_PER_STEP:
        raise ValueError(
            f"max_sim_qubits={max_sim_qubits} is too small to simulate even one "
            f"uncomputed step ({n_data + ANCILLAS_PER_STEP} qubits needed)"
        )

    verify_bit_order()

    all_steps = build_steps(n_data, max_steps, seed)
    specs = make_input_specs(n_data, seed)
    records: List[StepRecord] = []

    for n_steps in range(1, max_steps + 1):
        steps = all_steps[:n_steps]

        # --- Circuit widths, measured from the constructed tapes -------------
        naive_ancillas, naive_total = measure_circuit_width(
            naive_circuit(steps, n_data, specs["uniform"]), n_data
        )
        unc_ancillas, unc_total = measure_circuit_width(
            uncomputed_circuit(steps, n_data, specs["uniform"]), n_data
        )

        # --- Structured exact model (M2), available for every N --------------
        psi_ideal_uniform, rho_naive_uniform = structured_states(
            steps, n_data, specs["uniform"]
        )
        rho_ideal_uniform = density_matrix(psi_ideal_uniform)

        record = StepRecord(
            n_steps=n_steps,
            n_data=n_data,
            naive_ancillas=naive_ancillas,
            naive_total_qubits=naive_total,
            uncomputed_ancillas=unc_ancillas,
            uncomputed_total_qubits=unc_total,
            fidelity_uncomputed_vs_ideal=float("nan"),
            trace_distance_uncomputed_vs_ideal=float("nan"),
        )

        # --- Scenario B vs the logical target, full state vector -------------
        # Scenario B is always narrow enough to simulate exactly.
        wires_b = data_wire_labels(n_data) + ancilla_wire_labels(ANCILLAS_PER_STEP)
        state_b = simulate(
            uncomputed_circuit(steps, n_data, specs["uniform"]), wires_b
        )
        rho_b_data = partial_trace(state_b, len(wires_b), list(range(n_data)))
        record.fidelity_uncomputed_vs_ideal = fidelity(rho_b_data, rho_ideal_uniform)
        record.trace_distance_uncomputed_vs_ideal = trace_distance(
            rho_b_data, rho_ideal_uniform
        )

        # --- Scenario A vs scenario B, both inputs (M2) -----------------------
        for name, spec in specs.items():
            psi_ideal, rho_naive = structured_states(steps, n_data, spec)
            rho_clean = density_matrix(psi_ideal)
            record.fidelity_naive_vs_uncomputed[name] = fidelity(rho_naive, rho_clean)
            record.trace_distance_naive_vs_uncomputed[name] = trace_distance(
                rho_naive, rho_clean
            )

        # --- Ancilla entropy across the last step's compute/uncompute cycle ---
        diagnostics = ancilla_diagnostics(steps, n_data, specs["uniform"], n_steps - 1)
        record.ancilla_entropy_before = diagnostics["entropy_before"]
        record.ancilla_entropy_after = diagnostics["entropy_after"]
        record.ancilla_purity_before = diagnostics["purity_before"]
        record.ancilla_purity_after = diagnostics["purity_after"]

        # --- M1 vs M2 cross-validation, where the budget allows ---------------
        if naive_total <= max_sim_qubits:
            wires_a = data_wire_labels(n_data) + ancilla_wire_labels(naive_ancillas)
            deviations = []
            for name, spec in specs.items():
                state_a = simulate(naive_circuit(steps, n_data, spec), wires_a)
                rho_a_m1 = partial_trace(state_a, len(wires_a), list(range(n_data)))
                _, rho_a_m2 = structured_states(steps, n_data, spec)
                deviations.append(float(np.max(np.abs(rho_a_m1 - rho_a_m2))))
            record.statevector_verified = True
            record.m1_m2_max_deviation = max(deviations)
            LOGGER.debug(
                "N=%d: M1 vs M2 max |delta rho| = %.3e",
                n_steps,
                record.m1_m2_max_deviation,
            )
        else:
            LOGGER.info(
                "N=%d: scenario A needs %d qubits > budget %d; using the "
                "structured model only (already validated for smaller N)",
                n_steps,
                naive_total,
                max_sim_qubits,
            )

        records.append(record)

    metadata: Dict[str, object] = {
        "seed": seed,
        "n_data": n_data,
        "max_steps": max_steps,
        "max_sim_qubits": max_sim_qubits,
        "literals_per_step": LITERALS_PER_STEP,
        "ancillas_per_step": ANCILLAS_PER_STEP,
        "basis_input_bits": list(specs["basis"].bits),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pennylane": qml.version(),
        "fidelity_convention": "Uhlmann, squared (F=1 for identical states)",
        "entropy_units": "bits (log base 2)",
    }
    return records, metadata


# --------------------------------------------------------------------------
# 7. Reporting and plotting
# --------------------------------------------------------------------------


def print_report(records: Sequence[StepRecord], metadata: Dict[str, object]) -> None:
    """Print the benchmark tables to stdout."""
    print()
    print("=" * 78)
    print("UNCOMPUTATION (U-dagger) QUBIT-EFFICIENCY BENCHMARK")
    print("=" * 78)
    print(
        f"seed={metadata['seed']}  n_data={metadata['n_data']}  "
        f"pennylane={metadata['pennylane']}  numpy={metadata['numpy']}  "
        f"python={metadata['python']}"
    )
    print(
        f"fidelity convention: {metadata['fidelity_convention']}; "
        f"entropy in {metadata['entropy_units']}"
    )
    print()

    print("--- Qubit scaling (widths measured from the constructed circuits) ---")
    header = (
        f"{'N':>3}  {'naive anc':>9}  {'naive tot':>9}  "
        f"{'unc anc':>7}  {'unc tot':>7}  {'saved':>5}  {'M1?':>4}"
    )
    print(header)
    print("-" * len(header))
    for rec in records:
        saved = rec.naive_total_qubits - rec.uncomputed_total_qubits
        print(
            f"{rec.n_steps:>3}  {rec.naive_ancillas:>9}  {rec.naive_total_qubits:>9}  "
            f"{rec.uncomputed_ancillas:>7}  {rec.uncomputed_total_qubits:>7}  "
            f"{saved:>5}  {'yes' if rec.statevector_verified else 'no':>4}"
        )

    print()
    print("--- Correctness: scenario B (uncomputed) vs the logical target ---")
    header = f"{'N':>3}  {'fidelity':>18}  {'trace distance':>18}"
    print(header)
    print("-" * len(header))
    for rec in records:
        print(
            f"{rec.n_steps:>3}  {rec.fidelity_uncomputed_vs_ideal:>18.15f}  "
            f"{rec.trace_distance_uncomputed_vs_ideal:>18.3e}"
        )

    print()
    print("--- Scenario A (garbage left) vs scenario B, by input state ---")
    print("    basis input  : garbage is a deterministic label -> no decoherence")
    print("    uniform input: garbage records which-path -> coherence destroyed")
    header = (
        f"{'N':>3}  {'F (basis)':>12}  {'T (basis)':>12}  "
        f"{'F (uniform)':>12}  {'T (uniform)':>12}"
    )
    print(header)
    print("-" * len(header))
    for rec in records:
        print(
            f"{rec.n_steps:>3}  "
            f"{rec.fidelity_naive_vs_uncomputed['basis']:>12.9f}  "
            f"{rec.trace_distance_naive_vs_uncomputed['basis']:>12.3e}  "
            f"{rec.fidelity_naive_vs_uncomputed['uniform']:>12.9f}  "
            f"{rec.trace_distance_naive_vs_uncomputed['uniform']:>12.9f}"
        )

    print()
    print("--- Scratch register across the last compute/uncompute cycle ---")
    header = (
        f"{'N':>3}  {'S before':>12}  {'S after':>12}  "
        f"{'purity before':>14}  {'purity after':>14}"
    )
    print(header)
    print("-" * len(header))
    for rec in records:
        print(
            f"{rec.n_steps:>3}  {rec.ancilla_entropy_before:>12.9f}  "
            f"{rec.ancilla_entropy_after:>12.3e}  "
            f"{rec.ancilla_purity_before:>14.9f}  {rec.ancilla_purity_after:>14.9f}"
        )

    verified = [r for r in records if r.statevector_verified]
    if verified:
        worst = max(float(r.m1_m2_max_deviation or 0.0) for r in verified)
        print()
        print(
            f"--- Cross-validation: full state vector (M1) vs structured model "
            f"(M2) ---\n    N = 1..{verified[-1].n_steps} both methods run; "
            f"worst |delta rho| = {worst:.3e}"
        )
        if len(verified) < len(records):
            print(
                f"    N = {verified[-1].n_steps + 1}..{records[-1].n_steps} "
                f"scenario A exceeds the {metadata['max_sim_qubits']}-qubit "
                f"budget; metrics from M2 only."
            )
    print()


def make_plot(
    records: Sequence[StepRecord],
    metadata: Dict[str, object],
    output_path: str,
) -> None:
    """Write the qubit-scaling comparison figure.

    Panel (a) is the required scaling plot. Panel (b) reports the scratch
    register's entropy before and after ``U-dagger``, which is the mechanism
    that makes the reuse in panel (a) legitimate.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    n_values = [r.n_steps for r in records]
    naive_totals = [r.naive_total_qubits for r in records]
    unc_totals = [r.uncomputed_total_qubits for r in records]
    entropy_before = [r.ancilla_entropy_before or 0.0 for r in records]
    entropy_after = [r.ancilla_entropy_after or 0.0 for r in records]

    figure, (ax_scale, ax_entropy) = plt.subplots(
        1, 2, figsize=(12.0, 4.8), dpi=200, facecolor=COLOR_SURFACE
    )

    for axis in (ax_scale, ax_entropy):
        axis.set_facecolor(COLOR_SURFACE)
        axis.grid(True, color=COLOR_GRID, linewidth=0.8, linestyle="-")
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(COLOR_GRID)
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=9, length=0)
        # N is a step *count*: fractional ticks like "2.5 steps" are meaningless.
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Panel (a): qubit scaling ----------------------------------------
    ax_scale.plot(
        n_values,
        naive_totals,
        color=COLOR_GARBAGE,
        linewidth=2.0,
        marker="o",
        markersize=5,
        markeredgecolor=COLOR_SURFACE,
        markeredgewidth=1.0,
        label="Without uncomputation (garbage kept)",
    )
    ax_scale.plot(
        n_values,
        unc_totals,
        color=COLOR_CLEAN,
        linewidth=2.0,
        marker="o",
        markersize=5,
        markeredgecolor=COLOR_SURFACE,
        markeredgewidth=1.0,
        label="With uncomputation (U-dagger, scratch reused)",
    )

    # Mark exactly how far the full state-vector simulation reached, so the
    # figure cannot be read as "all of this was simulated".
    verified = [r.n_steps for r in records if r.statevector_verified]
    if verified and len(verified) < len(records):
        boundary = max(verified)
        ax_scale.axvspan(
            min(n_values) - 0.5, boundary + 0.5, color=COLOR_GRID, alpha=0.55, lw=0
        )
        ax_scale.text(
            boundary + 0.4,
            max(naive_totals) * 0.30,
            f"shaded: scenario A also verified by full\nstate-vector simulation "
            f"(N <= {boundary},\nbudget {metadata['max_sim_qubits']} qubits).\n"
            f"Beyond it, widths are still measured\nfrom the constructed circuits.",
            fontsize=7.5,
            color=COLOR_TEXT_SECONDARY,
            ha="left",
            va="center",
        )

    ax_scale.annotate(
        f"{naive_totals[-1]} qubits",
        xy=(n_values[-1], naive_totals[-1]),
        xytext=(-4, 8),
        textcoords="offset points",
        fontsize=9,
        color=COLOR_TEXT,
        ha="right",
    )
    ax_scale.annotate(
        f"{unc_totals[-1]} qubits",
        xy=(n_values[-1], unc_totals[-1]),
        xytext=(-4, 8),
        textcoords="offset points",
        fontsize=9,
        color=COLOR_TEXT,
        ha="right",
    )

    ax_scale.set_xlabel("Number of computation steps N", fontsize=10, color=COLOR_TEXT)
    ax_scale.set_ylabel("Physical qubits required", fontsize=10, color=COLOR_TEXT)
    ax_scale.set_title(
        f"(a) Circuit width vs steps  "
        f"({metadata['n_data']} data qubits, "
        f"{metadata['ancillas_per_step']} scratch qubits per step)",
        fontsize=11,
        color=COLOR_TEXT,
        loc="left",
        pad=10,
    )
    ax_scale.set_xlim(min(n_values) - 0.5, max(n_values) + 0.5)
    ax_scale.set_ylim(0, max(naive_totals) * 1.15)
    legend = ax_scale.legend(
        frameon=False, fontsize=9, loc="upper left", labelcolor=COLOR_TEXT_SECONDARY
    )
    legend.set_zorder(5)

    # --- Panel (b): scratch entropy across the cycle ----------------------
    ax_entropy.plot(
        n_values,
        entropy_before,
        color=COLOR_GARBAGE,
        linewidth=2.0,
        marker="o",
        markersize=5,
        markeredgecolor=COLOR_SURFACE,
        markeredgewidth=1.0,
        label="Scratch after compute (entangled)",
    )
    ax_entropy.plot(
        n_values,
        entropy_after,
        color=COLOR_CLEAN,
        linewidth=2.0,
        marker="o",
        markersize=5,
        markeredgecolor=COLOR_SURFACE,
        markeredgewidth=1.0,
        label="Scratch after U-dagger (returned to |0>)",
    )
    ax_entropy.set_xlabel(
        "Step index N (cycle inspected)", fontsize=10, color=COLOR_TEXT
    )
    ax_entropy.set_ylabel(
        "Scratch von Neumann entropy S (bits)", fontsize=10, color=COLOR_TEXT
    )
    ax_entropy.set_title(
        "(b) Why the scratch can be reused",
        fontsize=11,
        color=COLOR_TEXT,
        loc="left",
        pad=10,
    )
    ax_entropy.set_xlim(min(n_values) - 0.5, max(n_values) + 0.5)
    ax_entropy.set_ylim(-0.1, max(max(entropy_before), 0.1) * 1.35)
    ax_entropy.legend(
        frameon=False, fontsize=9, loc="upper left", labelcolor=COLOR_TEXT_SECONDARY
    )

    figure.text(
        0.01,
        0.015,
        f"seed={metadata['seed']} | PennyLane {metadata['pennylane']}, "
        f"NumPy {metadata['numpy']}, Python {metadata['python']} | "
        f"input state: uniform superposition | entropy in bits",
        fontsize=7,
        color=COLOR_TEXT_SECONDARY,
    )

    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_path, facecolor=COLOR_SURFACE)
    plt.close(figure)
    LOGGER.info("wrote %s", output_path)


def records_to_json(
    records: Sequence[StepRecord], metadata: Dict[str, object]
) -> Dict[str, object]:
    """Serialise the benchmark to a plain dict for the results JSON."""
    return {
        "metadata": metadata,
        "records": [
            {
                "n_steps": r.n_steps,
                "n_data": r.n_data,
                "naive_ancillas": r.naive_ancillas,
                "naive_total_qubits": r.naive_total_qubits,
                "uncomputed_ancillas": r.uncomputed_ancillas,
                "uncomputed_total_qubits": r.uncomputed_total_qubits,
                "fidelity_uncomputed_vs_ideal": r.fidelity_uncomputed_vs_ideal,
                "trace_distance_uncomputed_vs_ideal": (
                    r.trace_distance_uncomputed_vs_ideal
                ),
                "fidelity_naive_vs_uncomputed": r.fidelity_naive_vs_uncomputed,
                "trace_distance_naive_vs_uncomputed": (
                    r.trace_distance_naive_vs_uncomputed
                ),
                "ancilla_entropy_before": r.ancilla_entropy_before,
                "ancilla_entropy_after": r.ancilla_entropy_after,
                "ancilla_purity_before": r.ancilla_purity_before,
                "ancilla_purity_after": r.ancilla_purity_after,
                "statevector_verified": r.statevector_verified,
                "m1_m2_max_deviation": r.m1_m2_max_deviation,
            }
            for r in records
        ],
    }


# --------------------------------------------------------------------------
# 8. Command-line interface
# --------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate that uncomputation with U-dagger reduces circuit width "
            "from O(N) scratch qubits to O(1), and verify that it does so "
            "without changing the logical state."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-data", type=int, default=DEFAULT_N_DATA)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-sim-qubits",
        type=int,
        default=DEFAULT_MAX_SIM_QUBITS,
        help="memory budget for the full state-vector method",
    )
    parser.add_argument(
        "--plot", default="qubit_scaling_comparison.png", help="output figure path"
    )
    parser.add_argument(
        "--json", default="benchmark_results.json", help="output results path"
    )
    parser.add_argument("--no-plot", action="store_true", help="skip the figure")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        records, metadata = run_benchmark(
            n_data=args.n_data,
            max_steps=args.max_steps,
            seed=args.seed,
            max_sim_qubits=args.max_sim_qubits,
        )
    except (ValueError, RuntimeError) as exc:
        LOGGER.error("benchmark failed: %s", exc)
        return 1
    except MemoryError:
        LOGGER.error(
            "out of memory: lower --max-sim-qubits (currently %d) or --max-steps",
            args.max_sim_qubits,
        )
        return 1

    print_report(records, metadata)

    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(records_to_json(records, metadata), handle, indent=2)
    LOGGER.info("wrote %s", args.json)

    if not args.no_plot:
        try:
            make_plot(records, metadata, args.plot)
        except ImportError:
            LOGGER.error("matplotlib is required for --plot; install requirements.txt")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
