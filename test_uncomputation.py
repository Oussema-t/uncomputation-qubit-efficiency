#!/usr/bin/env python3
"""Validation suite for the uncomputation demonstration.

Every check here was specified *before* the implementation was written (see the
VALIDATION block in the README) so that no success criterion is reverse-engineered
from the numbers that came out.

The suite covers:

  V1  Scenario B reproduces the logical target exactly (F -> 1, T -> 0).
  V2  The scratch register is entangled after compute and clean after U-dagger.
  V3  On a computational-basis input, scenario A and scenario B agree.
  V4  On a superposition input, they do NOT -- garbage decoheres the data.
  V5  The structured exact model (M2) agrees with the full state vector (M1).
  V6  Measured circuit widths scale as O(N) without cleanup and O(1) with it.

Plus: metric correctness against analytically known states, the claim that the
compute subroutine is genuinely not self-inverse, seed robustness, and error
handling.

Run with pytest if available, otherwise directly:

    pytest -v test_uncomputation.py
    python test_uncomputation.py
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pennylane as qml

from uncomputation_demo import (
    ANCILLAS_PER_STEP,
    LITERALS_PER_STEP,
    Step,
    InputSpec,
    ancilla_diagnostics,
    ancilla_wire_labels,
    basis_bit_table,
    build_steps,
    compute_step,
    data_wire_labels,
    density_matrix,
    evaluate_predicates,
    fidelity,
    logical_phases,
    make_input_specs,
    measure_circuit_width,
    naive_circuit,
    partial_trace,
    purity,
    run_benchmark,
    simulate,
    structured_states,
    trace_distance,
    uncomputed_circuit,
    verify_bit_order,
    von_neumann_entropy,
)

# Tolerances. EXACT_TOL is for quantities that are mathematically exact and
# limited only by float64 round-off; LOOSE_TOL is for the qualitative
# "clearly non-zero" thresholds in V2/V4.
EXACT_TOL = 1e-9
LOOSE_TOL = 1e-2

N_DATA = 4
SEED = 20240517
#: Scenario A costs 2**(N_DATA + 2N) amplitudes, so the full state-vector
#: cross-check in the test suite stops at 16 qubits to stay fast.
MAX_N_FULL_SIM = 6


# --------------------------------------------------------------------------
# Foundations: conventions and primitives the rest of the suite depends on
# --------------------------------------------------------------------------


def test_bit_order_convention_matches_pennylane() -> None:
    """The bit-string table must match qml.state()'s wire ordering.

    If this is wrong, every partial trace below is silently wrong too.
    """
    verify_bit_order()


def test_basis_bit_table_is_big_endian() -> None:
    table = basis_bit_table(3)
    assert table.shape == (8, 3)
    assert list(table[0]) == [0, 0, 0]
    assert list(table[1]) == [0, 0, 1]
    assert list(table[4]) == [1, 0, 0]
    assert list(table[7]) == [1, 1, 1]


def test_partial_trace_on_a_bell_state() -> None:
    """A Bell state's single-qubit marginal is maximally mixed: S = 1 bit."""
    bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    rho = partial_trace(bell, 2, [0])
    assert np.allclose(rho, np.eye(2) / 2, atol=EXACT_TOL)
    assert abs(von_neumann_entropy(rho) - 1.0) < EXACT_TOL
    assert abs(purity(rho) - 0.5) < EXACT_TOL


def test_partial_trace_on_a_product_state() -> None:
    """A product state's marginal is pure: S = 0."""
    plus_zero = np.array([1, 0, 1, 0], dtype=complex) / math.sqrt(2)
    rho = partial_trace(plus_zero, 2, [1])
    assert abs(von_neumann_entropy(rho)) < EXACT_TOL
    assert abs(purity(rho) - 1.0) < EXACT_TOL


def test_metrics_on_analytically_known_states() -> None:
    """Fidelity and trace distance against cases with closed-form answers."""
    zero = density_matrix(np.array([1, 0], dtype=complex))
    one = density_matrix(np.array([0, 1], dtype=complex))
    plus = density_matrix(np.array([1, 1], dtype=complex) / math.sqrt(2))
    mixed = np.eye(2, dtype=complex) / 2

    # Identical pure states.
    assert abs(fidelity(zero, zero) - 1.0) < EXACT_TOL
    assert abs(trace_distance(zero, zero)) < EXACT_TOL

    # Orthogonal pure states.
    assert abs(fidelity(zero, one)) < EXACT_TOL
    assert abs(trace_distance(zero, one) - 1.0) < EXACT_TOL

    # |<0|+>|**2 = 1/2 in the squared convention this module uses.
    assert abs(fidelity(zero, plus) - 0.5) < EXACT_TOL
    # Trace distance between |0> and |+> is 1/sqrt(2).
    assert abs(trace_distance(zero, plus) - 1.0 / math.sqrt(2)) < EXACT_TOL

    # Pure vs maximally mixed: F = 1/2, T = 1/2.
    assert abs(fidelity(zero, mixed) - 0.5) < EXACT_TOL
    assert abs(trace_distance(zero, mixed) - 0.5) < EXACT_TOL


def test_fidelity_never_exceeds_one() -> None:
    """Fidelity is bounded above by 1; round-off must not push it past that."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        vector = rng.normal(size=8) + 1j * rng.normal(size=8)
        vector /= np.linalg.norm(vector)
        rho = density_matrix(vector)
        assert fidelity(rho, rho) <= 1.0
        assert fidelity(rho, rho) > 1.0 - EXACT_TOL


# --------------------------------------------------------------------------
# The claim that U-dagger is a real inverse, not a repeat of U
# --------------------------------------------------------------------------


def test_compute_step_is_not_self_inverse() -> None:
    """U_k != U_k-dagger, so the demo genuinely exercises an adjoint.

    The two Toffolis do not commute (the scratch qubit t is the target of the
    first and a control of the second), so reversing them is a real change.
    Asserted numerically rather than argued in a comment.
    """
    step = Step(qubits=(0, 1, 2), negations=(False, True, False), theta=0.7)
    data = data_wire_labels(N_DATA)
    anc_t, anc_r = ancilla_wire_labels(ANCILLAS_PER_STEP)
    order = data + [anc_t, anc_r]

    forward = qml.matrix(compute_step, wire_order=order)(step, data, anc_t, anc_r)
    inverse = qml.matrix(qml.adjoint(compute_step), wire_order=order)(
        step, data, anc_t, anc_r
    )

    assert not np.allclose(forward, inverse), "U_k is self-inverse; demo is trivial"
    # But it must be a true inverse.
    assert np.allclose(forward @ inverse, np.eye(forward.shape[0]), atol=EXACT_TOL)


def test_compute_step_writes_the_expected_predicate() -> None:
    """U_k must actually compute the conjunction the classical model claims.

    Without this, the "logical target" would be defined by the circuit rather
    than checked against an independent reference.
    """
    steps = build_steps(N_DATA, 1, SEED)
    step = steps[0]
    predicates = evaluate_predicates(steps, N_DATA)[0]
    bits = basis_bit_table(N_DATA)
    data = data_wire_labels(N_DATA)
    anc_t, anc_r = ancilla_wire_labels(ANCILLAS_PER_STEP)
    wires = data + [anc_t, anc_r]

    for index in range(2**N_DATA):

        def circuit(idx: int = index) -> None:
            for bit, wire in zip(bits[idx], data):
                if bit:
                    qml.PauliX(wires=wire)
            compute_step(step, data, anc_t, anc_r)

        state = simulate(circuit, wires)
        peak = int(np.argmax(np.abs(state)))
        # Scratch qubit r is the last wire; its value is the predicate.
        r_value = peak & 1
        assert r_value == int(predicates[index]), (
            f"basis state {index}: circuit wrote r={r_value}, "
            f"classical model says {int(predicates[index])}"
        )


# --------------------------------------------------------------------------
# V2 -- entanglement before, cleanliness after
# --------------------------------------------------------------------------


def test_v2_scratch_is_entangled_before_and_clean_after() -> None:
    """S > 0 with garbage present; S = 0 and purity = 1 after U-dagger."""
    steps = build_steps(N_DATA, 8, SEED)
    specs = make_input_specs(N_DATA, SEED)

    for index in range(len(steps)):
        diagnostics = ancilla_diagnostics(steps, N_DATA, specs["uniform"], index)
        assert diagnostics["entropy_before"] > 0.1, (
            f"step {index}: scratch entropy {diagnostics['entropy_before']:.3e} "
            f"is not clearly non-zero; nothing was entangled"
        )
        assert diagnostics["entropy_after"] < EXACT_TOL, (
            f"step {index}: scratch entropy {diagnostics['entropy_after']:.3e} "
            f"after U-dagger; the inverse did not clean up"
        )
        assert diagnostics["purity_before"] < 1.0 - LOOSE_TOL
        assert abs(diagnostics["purity_after"] - 1.0) < EXACT_TOL


def test_scratch_returns_to_the_zero_state_exactly() -> None:
    """Stronger than S = 0: the scratch must be |00>, not merely unentangled."""
    steps = build_steps(N_DATA, 5, SEED)
    specs = make_input_specs(N_DATA, SEED)
    wires = data_wire_labels(N_DATA) + ancilla_wire_labels(ANCILLAS_PER_STEP)

    state = simulate(uncomputed_circuit(steps, N_DATA, specs["uniform"]), wires)
    tensor = state.reshape([2] * len(wires))
    # Every amplitude with a scratch qubit set to 1 must vanish.
    leaked = float(np.sum(np.abs(tensor) ** 2)) - float(
        np.sum(np.abs(tensor[..., 0, 0]) ** 2)
    )
    assert leaked < EXACT_TOL, f"population {leaked:.3e} left outside |00> scratch"


# --------------------------------------------------------------------------
# V1 -- scenario B reproduces the logical target
# --------------------------------------------------------------------------


def test_v1_uncomputed_circuit_matches_the_logical_target() -> None:
    """F = 1 and T = 0 against an independently computed reference state."""
    max_steps = 20
    all_steps = build_steps(N_DATA, max_steps, SEED)
    specs = make_input_specs(N_DATA, SEED)
    wires = data_wire_labels(N_DATA) + ancilla_wire_labels(ANCILLAS_PER_STEP)

    for n_steps in range(1, max_steps + 1):
        steps = all_steps[:n_steps]
        for name, spec in specs.items():
            state = simulate(uncomputed_circuit(steps, N_DATA, spec), wires)
            rho_actual = partial_trace(state, len(wires), list(range(N_DATA)))

            # Reference: classical phase accumulation, no circuit involved.
            amplitudes = np.zeros(2**N_DATA, dtype=complex)
            if spec.kind == "uniform":
                amplitudes[:] = 1.0 / math.sqrt(2**N_DATA)
            else:
                index = 0
                for bit in spec.bits:
                    index = (index << 1) | bit
                amplitudes[index] = 1.0
            reference = density_matrix(
                amplitudes * np.exp(1j * logical_phases(steps, N_DATA))
            )

            fid = fidelity(rho_actual, reference)
            dist = trace_distance(rho_actual, reference)
            assert fid > 1.0 - EXACT_TOL, f"N={n_steps} {name}: fidelity {fid}"
            assert dist < EXACT_TOL, f"N={n_steps} {name}: trace distance {dist}"


# --------------------------------------------------------------------------
# V3 / V4 -- when garbage is harmless, and when it is not
# --------------------------------------------------------------------------


def test_v3_basis_input_makes_the_two_scenarios_agree() -> None:
    """On a basis input the garbage is a deterministic label -> no damage."""
    all_steps = build_steps(N_DATA, 12, SEED)
    spec = make_input_specs(N_DATA, SEED)["basis"]

    for n_steps in range(1, 13):
        psi_clean, rho_naive = structured_states(all_steps[:n_steps], N_DATA, spec)
        rho_clean = density_matrix(psi_clean)
        assert fidelity(rho_naive, rho_clean) > 1.0 - EXACT_TOL
        assert trace_distance(rho_naive, rho_clean) < EXACT_TOL


def test_v4_superposition_input_breaks_the_naive_scenario() -> None:
    """The headline: with garbage kept, a superposition input is decohered.

    This is the claim the task statement's "fidelity approx 1" does NOT cover.
    Uncomputation is required for correctness here, not merely for qubit count.
    """
    all_steps = build_steps(N_DATA, 12, SEED)
    spec = make_input_specs(N_DATA, SEED)["uniform"]

    for n_steps in range(1, 13):
        psi_clean, rho_naive = structured_states(all_steps[:n_steps], N_DATA, spec)
        rho_clean = density_matrix(psi_clean)
        fid = fidelity(rho_naive, rho_clean)
        dist = trace_distance(rho_naive, rho_clean)
        assert fid < 1.0 - LOOSE_TOL, (
            f"N={n_steps}: fidelity {fid:.6f} -- garbage was expected to "
            f"decohere the data register but did not"
        )
        assert dist > LOOSE_TOL, f"N={n_steps}: trace distance {dist:.6f}"


def test_naive_damage_is_monotone_non_decreasing_in_n() -> None:
    """Each extra un-cleaned step can only add to the which-path record."""
    all_steps = build_steps(N_DATA, 12, SEED)
    spec = make_input_specs(N_DATA, SEED)["uniform"]

    previous = math.inf
    for n_steps in range(1, 13):
        psi_clean, rho_naive = structured_states(all_steps[:n_steps], N_DATA, spec)
        fid = fidelity(rho_naive, density_matrix(psi_clean))
        assert fid <= previous + EXACT_TOL, (
            f"N={n_steps}: fidelity rose from {previous:.9f} to {fid:.9f}; "
            f"adding garbage cannot restore coherence"
        )
        previous = fid


# --------------------------------------------------------------------------
# V5 -- the two simulation methods must agree
# --------------------------------------------------------------------------


def test_v5_structured_model_matches_full_statevector() -> None:
    """M2 is only usable at large N because it matches M1 at small N."""
    all_steps = build_steps(N_DATA, MAX_N_FULL_SIM, SEED)
    specs = make_input_specs(N_DATA, SEED)

    for n_steps in range(1, MAX_N_FULL_SIM + 1):
        steps = all_steps[:n_steps]
        n_ancillas = ANCILLAS_PER_STEP * n_steps
        wires = data_wire_labels(N_DATA) + ancilla_wire_labels(n_ancillas)

        for name, spec in specs.items():
            state = simulate(naive_circuit(steps, N_DATA, spec), wires)
            rho_m1 = partial_trace(state, len(wires), list(range(N_DATA)))
            _, rho_m2 = structured_states(steps, N_DATA, spec)
            deviation = float(np.max(np.abs(rho_m1 - rho_m2)))
            assert deviation < EXACT_TOL, (
                f"N={n_steps} {name}: structured model deviates from the full "
                f"state vector by {deviation:.3e}"
            )


# --------------------------------------------------------------------------
# V6 -- measured circuit widths
# --------------------------------------------------------------------------


def test_v6_measured_widths_scale_as_expected() -> None:
    """Widths are read off the constructed tapes, then checked against O(N)/O(1)."""
    all_steps = build_steps(N_DATA, 20, SEED)
    spec = make_input_specs(N_DATA, SEED)["uniform"]

    naive_widths: List[int] = []
    uncomputed_widths: List[int] = []
    for n_steps in range(1, 21):
        steps = all_steps[:n_steps]
        _, naive_total = measure_circuit_width(
            naive_circuit(steps, N_DATA, spec), N_DATA
        )
        _, unc_total = measure_circuit_width(
            uncomputed_circuit(steps, N_DATA, spec), N_DATA
        )
        naive_widths.append(naive_total)
        uncomputed_widths.append(unc_total)

        assert naive_total == N_DATA + ANCILLAS_PER_STEP * n_steps
        assert unc_total == N_DATA + ANCILLAS_PER_STEP

    # Linear growth with slope ANCILLAS_PER_STEP, versus a flat line.
    naive_deltas = np.diff(naive_widths)
    uncomputed_deltas = np.diff(uncomputed_widths)
    assert np.all(naive_deltas == ANCILLAS_PER_STEP)
    assert np.all(uncomputed_deltas == 0)


# --------------------------------------------------------------------------
# Robustness: the result must not depend on one lucky seed
# --------------------------------------------------------------------------


def test_results_hold_across_multiple_seeds() -> None:
    """V1, V2 and V4 re-checked on seeds the demo does not default to."""
    for seed in (1, 2, 3, 12345):
        steps = build_steps(N_DATA, 6, seed)
        specs = make_input_specs(N_DATA, seed)
        wires = data_wire_labels(N_DATA) + ancilla_wire_labels(ANCILLAS_PER_STEP)

        # V1
        state = simulate(uncomputed_circuit(steps, N_DATA, specs["uniform"]), wires)
        rho = partial_trace(state, len(wires), list(range(N_DATA)))
        psi_clean, rho_naive = structured_states(steps, N_DATA, specs["uniform"])
        assert fidelity(rho, density_matrix(psi_clean)) > 1.0 - EXACT_TOL

        # V2
        diagnostics = ancilla_diagnostics(steps, N_DATA, specs["uniform"], 0)
        assert diagnostics["entropy_before"] > 0.1
        assert diagnostics["entropy_after"] < EXACT_TOL

        # V4
        assert fidelity(rho_naive, density_matrix(psi_clean)) < 1.0 - LOOSE_TOL


def test_results_hold_for_a_wider_data_register() -> None:
    """Nothing above should depend on n_data being exactly 4."""
    n_data = 6
    steps = build_steps(n_data, 5, SEED)
    specs = make_input_specs(n_data, SEED)
    wires = data_wire_labels(n_data) + ancilla_wire_labels(ANCILLAS_PER_STEP)

    state = simulate(uncomputed_circuit(steps, n_data, specs["uniform"]), wires)
    rho = partial_trace(state, len(wires), list(range(n_data)))
    psi_clean, rho_naive = structured_states(steps, n_data, specs["uniform"])
    assert fidelity(rho, density_matrix(psi_clean)) > 1.0 - EXACT_TOL
    assert fidelity(rho_naive, density_matrix(psi_clean)) < 1.0 - LOOSE_TOL


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


def _expect_value_error(callable_obj, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    try:
        callable_obj(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"{callable_obj.__name__} did not raise ValueError")


def test_invalid_configurations_raise_clear_errors() -> None:
    # Too few data qubits for distinct literals.
    _expect_value_error(build_steps, LITERALS_PER_STEP - 1, 3, SEED)
    # Negative step count.
    _expect_value_error(build_steps, N_DATA, -1, SEED)
    # Duplicate literals in a hand-built step.
    _expect_value_error(Step, qubits=(0, 0, 1), negations=(False, False, False), theta=0.1)
    # Wrong literal count.
    _expect_value_error(Step, qubits=(0, 1), negations=(False, False), theta=0.1)
    # Unknown input kind.
    _expect_value_error(InputSpec, kind="haar")
    # Simulation budget too small to run even one step.
    _expect_value_error(run_benchmark, N_DATA, 3, SEED, 2)
    # Partial trace with out-of-range indices.
    _expect_value_error(partial_trace, np.array([1, 0, 0, 0], dtype=complex), 2, [5])
    # Mismatched shapes in the metrics.
    _expect_value_error(fidelity, np.eye(2, dtype=complex), np.eye(4, dtype=complex))
    _expect_value_error(
        trace_distance, np.eye(2, dtype=complex), np.eye(4, dtype=complex)
    )


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------


def test_benchmark_runs_end_to_end_and_self_validates() -> None:
    """The full runner must produce internally consistent, verified records."""
    records, metadata = run_benchmark(
        n_data=N_DATA, max_steps=8, seed=SEED, max_sim_qubits=16
    )
    assert len(records) == 8
    assert metadata["seed"] == SEED

    verified = [r for r in records if r.statevector_verified]
    assert verified, "no N was cross-validated against the full state vector"
    for record in verified:
        assert record.m1_m2_max_deviation is not None
        assert record.m1_m2_max_deviation < EXACT_TOL

    for record in records:
        assert record.fidelity_uncomputed_vs_ideal > 1.0 - EXACT_TOL
        assert record.trace_distance_uncomputed_vs_ideal < EXACT_TOL
        assert record.fidelity_naive_vs_uncomputed["basis"] > 1.0 - EXACT_TOL
        assert record.fidelity_naive_vs_uncomputed["uniform"] < 1.0 - LOOSE_TOL
        assert (record.ancilla_entropy_before or 0.0) > 0.1
        assert (record.ancilla_entropy_after or 1.0) < EXACT_TOL


def _main() -> int:
    """Standalone runner, so the suite works without pytest installed."""
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n      {exc}")
        except Exception as exc:  # noqa: BLE001 - report, do not swallow
            failures += 1
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
