#!/usr/bin/env python3
"""Validation suite for the QAOA shift-scheduling study.

Criteria fixed before implementation:

  W1  the uncomputed cost layer implements exp(-i gamma C(x)) exactly
  W2  the scratch returns to |0> after each clause's U-dagger
  W3  measured widths: n_vars + 2*n_clauses naive, n_vars + 2 uncomputed
  W4  solution quality vs a uniform-random baseline  (reported, not asserted --
      see the note on test_w4 below)
  W5  the naive dephasing model matches a full state vector at p = 1
  W6  results are reported with their spread across optimiser restarts
  W7  the naive dephasing model is ALSO exact at p = 2 and p = 3, checked on a
      shrunk instance where the full multi-layer circuit is simulable -- this
      closes the coverage gap that the real 33/45-qubit problem cannot close
      directly, and it is the composition step the headline result depends on

Plus: the classical problem encoding, the fast optimiser path against the real
circuit, mixer unitarity, baseline sanity, and error handling.

    python test_qaoa_scheduling.py
    pytest -v test_qaoa_scheduling.py
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from qaoa_scheduling import (
    ANCILLAS_PER_STEP,
    HARD_PENALTY,
    SHIFT_COST,
    Baselines,
    SchedulingProblem,
    ancilla_wire_labels,
    apply_mixer_to_state,
    build_problem,
    clause_violations,
    compute_baselines,
    cost_vector,
    cross_check_naive_model,
    data_wire_labels,
    describe_assignment,
    measure_widths,
    mixer_matrix,
    normalised_score,
    optimise,
    probabilities_naive,
    probabilities_uncomputed,
    probabilities_uncomputed_circuit,
    qaoa_circuit_naive,
    scratch_diagnostics,
    simulate,
    verify_cost_layer,
)
from uncomputation_demo import Step, basis_bit_table

EXACT_TOL = 1e-9
SEED = 20240517


# --------------------------------------------------------------------------
# The classical encoding
# --------------------------------------------------------------------------


def test_problem_shape() -> None:
    problem = build_problem()
    assert problem.n_vars == 9
    # 3 coverage clauses + 3 no-burnout clauses
    assert len(problem.clauses) == 6
    assert len(problem.clause_labels) == len(problem.clauses)
    for clause in problem.clauses:
        assert len(clause.qubits) == 3
        assert len(set(clause.qubits)) == 3


def test_clause_semantics_match_hand_computation() -> None:
    """Check the encoding against assignments reasoned out by hand.

    Without this, the circuit could faithfully implement the wrong problem.
    """
    problem = build_problem()
    violations = clause_violations(problem)
    bits = basis_bit_table(problem.n_vars)

    # The empty roster: nobody works. All 3 coverage clauses violated,
    # no burnout possible.
    empty = 0
    assert violations[:3, empty].sum() == 3, "empty roster must break all coverage"
    assert violations[3:, empty].sum() == 0, "empty roster cannot cause burnout"

    # The full roster: everyone works everything. Coverage fine, all 3 burnt out.
    full = 2**problem.n_vars - 1
    assert violations[:3, full].sum() == 0
    assert violations[3:, full].sum() == 3

    # A hand-built feasible roster: S1 -> shift1, S2 -> shift2, S3 -> shift3.
    index = 0
    for staff in range(3):
        for shift in range(3):
            index = (index << 1) | (1 if staff == shift else 0)
    assert violations[:, index].sum() == 0, (
        f"diagonal roster should be feasible, got "
        f"{describe_assignment(problem, index)}"
    )
    assert bits[index].sum() == 3


def test_cost_and_optimum_are_consistent() -> None:
    problem = build_problem()
    costs = cost_vector(problem)
    baselines = compute_baselines(problem)

    # Cheapest feasible roster covers 3 shifts with 3 assignments, no penalty.
    assert abs(baselines.optimum - 3.0 * problem.shift_cost) < EXACT_TOL
    # Every reported optimum must actually be feasible and actually be optimal.
    violations = clause_violations(problem)
    for index in baselines.optimum_indices:
        assert violations[:, index].sum() == 0
        assert abs(costs[index] - baselines.optimum) < EXACT_TOL
    assert baselines.optimum <= baselines.random_mean <= baselines.worst
    assert 0.0 < baselines.feasible_fraction < 1.0


def test_normalised_score_endpoints() -> None:
    problem = build_problem()
    baselines = compute_baselines(problem)
    assert abs(normalised_score(baselines.optimum, baselines) - 1.0) < EXACT_TOL
    assert abs(normalised_score(baselines.random_mean, baselines)) < EXACT_TOL
    # Worse than guessing must come out negative, not be clipped to zero.
    assert normalised_score(baselines.worst, baselines) < 0.0


# --------------------------------------------------------------------------
# W1 / W2 -- the cost layer and the scratch
# --------------------------------------------------------------------------


def test_w1_cost_layer_implements_the_intended_unitary() -> None:
    """The circuit must reproduce exp(-i gamma C(x)) from the classical model."""
    problem = build_problem()
    for gamma in (0.13, 0.37, 1.1, 2.9):
        deviation = verify_cost_layer(problem, gamma=gamma)
        assert deviation < EXACT_TOL, f"gamma={gamma}: deviation {deviation:.3e}"


def test_w2_scratch_is_clean_after_every_clause() -> None:
    problem = build_problem()
    for index in range(len(problem.clauses)):
        diagnostics = scratch_diagnostics(problem, gamma=0.37, clause_index=index)
        assert diagnostics["entropy_before"] > 0.1, (
            f"clause {index}: scratch never became entangled "
            f"(S = {diagnostics['entropy_before']:.3e})"
        )
        assert diagnostics["entropy_after"] < EXACT_TOL, (
            f"clause {index}: U-dagger failed to clean up "
            f"(S = {diagnostics['entropy_after']:.3e})"
        )
        assert abs(diagnostics["purity_after"] - 1.0) < EXACT_TOL


# --------------------------------------------------------------------------
# W3 -- measured widths
# --------------------------------------------------------------------------


def test_w3_measured_widths() -> None:
    problem = build_problem()
    for layers in (1, 2, 3):
        widths = measure_widths(problem, layers)
        assert widths["uncomputed"] == problem.n_vars + ANCILLAS_PER_STEP
        assert widths["naive"] == (
            problem.n_vars + ANCILLAS_PER_STEP * len(problem.clauses) * layers
        )
        assert widths["naive"] > widths["uncomputed"]

    # Uncomputed width must not grow with depth; naive must.
    flat = [measure_widths(problem, p)["uncomputed"] for p in (1, 2, 3)]
    grows = [measure_widths(problem, p)["naive"] for p in (1, 2, 3)]
    assert len(set(flat)) == 1
    assert grows[0] < grows[1] < grows[2]


# --------------------------------------------------------------------------
# W5 -- the naive dephasing model against a full state vector
# --------------------------------------------------------------------------


def test_w5_naive_model_matches_full_statevector() -> None:
    problem = build_problem()
    deviation = cross_check_naive_model(problem, gamma=0.37, beta=0.21)
    assert deviation is not None, "cross-check was skipped; raise the qubit budget"
    assert deviation < EXACT_TOL, f"model deviates by {deviation:.3e}"


def _small_instance() -> SchedulingProblem:
    """A 4-variable, 2-clause instance small enough to simulate at depth 3.

    The full naive circuit needs ``n_vars + 2 * n_clauses * p`` qubits, so the
    real 9-variable / 6-clause problem is 33 qubits at p=2 and 45 at p=3 -- out
    of reach. Here it is 12 and 16 qubits, fully simulable, exercising the
    identical ``probabilities_naive`` code path. The two clauses share a variable
    so their garbage classes are non-trivially correlated.
    """
    clauses = (
        Step(qubits=(0, 1, 2), negations=(False, False, False), theta=HARD_PENALTY),
        Step(qubits=(1, 2, 3), negations=(True, False, True), theta=HARD_PENALTY),
    )
    return SchedulingProblem(
        n_staff=4, n_shifts=1, clauses=clauses,
        clause_labels=("c0", "c1"), hard_penalty=HARD_PENALTY, shift_cost=SHIFT_COST,
    )


def _full_naive_data_probs(problem, gammas, betas):  # type: ignore[no-untyped-def]
    """Data-register outcome distribution from the FULL naive state vector."""
    per_layer = ANCILLAS_PER_STEP * len(problem.clauses)
    n_anc = per_layer * len(gammas)
    wires = data_wire_labels(problem.n_vars) + ancilla_wire_labels(n_anc)
    state = simulate(qaoa_circuit_naive(problem, gammas, betas), wires)
    tensor = state.reshape(2**problem.n_vars, 2**n_anc)
    return np.sum(np.abs(tensor) ** 2, axis=1)


def test_w7_naive_model_exact_at_depth_2_and_3() -> None:
    """The dephasing model must be exact at p=2 and p=3, not just p=1.

    This is the load-bearing composition step: the headline solution-quality
    result comes from p=2 and p=3, where the real circuit cannot be simulated, so
    the model's validity there is what the whole finding rests on. Checked on a
    shrunk instance where the full multi-layer circuit IS simulable.
    """
    problem = _small_instance()
    schedules = [([0.3, 0.5, 0.7][:p], [0.2, 0.15, 0.25][:p]) for p in (1, 2, 3)]
    for gammas, betas in schedules:
        model = probabilities_naive(problem, gammas, betas)
        exact = _full_naive_data_probs(problem, gammas, betas)
        deviation = float(np.max(np.abs(model - exact)))
        assert deviation < EXACT_TOL, (
            f"p={len(gammas)}: dephasing model deviates from the full state "
            f"vector by {deviation:.3e}"
        )


def test_w7_teeth_the_model_actually_dephases() -> None:
    """Guard against a vacuous pass: at these depths the naive (dephased) and
    uncomputed (coherent) distributions must genuinely differ, so the p=2/p=3
    agreement above is a real check of dephasing and not two identical trivia."""
    problem = _small_instance()
    for p in (2, 3):
        gammas, betas = [0.3, 0.5, 0.7][:p], [0.2, 0.15, 0.25][:p]
        naive = probabilities_naive(problem, gammas, betas)
        clean = probabilities_uncomputed(problem, gammas, betas)
        assert np.max(np.abs(naive - clean)) > 1e-2, (
            f"p={p}: naive and uncomputed distributions are identical; the "
            f"dephasing model is not exercising any dephasing"
        )


def test_fast_path_matches_the_real_circuit() -> None:
    """The optimiser's fast path must equal the actual PennyLane circuit.

    The fast path is only valid because uncomputation makes the scratch factor
    out. If that ever stopped being true, this is the test that would catch it.
    """
    problem = build_problem()
    for gammas, betas in (([0.37], [0.21]), ([0.5, 0.9], [0.3, 0.15])):
        fast = probabilities_uncomputed(problem, gammas, betas)
        exact = probabilities_uncomputed_circuit(problem, gammas, betas)
        assert np.max(np.abs(fast - exact)) < EXACT_TOL


def test_distributions_are_normalised() -> None:
    problem = build_problem()
    for probabilities in (
        probabilities_uncomputed(problem, [0.4], [0.25]),
        probabilities_naive(problem, [0.4], [0.25]),
        probabilities_uncomputed(problem, [0.4, 0.8], [0.25, 0.1]),
        probabilities_naive(problem, [0.4, 0.8], [0.25, 0.1]),
    ):
        assert abs(probabilities.sum() - 1.0) < EXACT_TOL
        assert np.all(probabilities >= -EXACT_TOL)


# --------------------------------------------------------------------------
# The mixer
# --------------------------------------------------------------------------


def test_mixer_is_unitary_and_matches_the_dense_form() -> None:
    for beta in (0.0, 0.3, 1.4):
        dense = mixer_matrix(4, beta)
        assert np.allclose(dense @ dense.conj().T, np.eye(16), atol=EXACT_TOL)

        rng = np.random.default_rng(0)
        vector = rng.normal(size=16) + 1j * rng.normal(size=16)
        vector /= np.linalg.norm(vector)
        assert np.allclose(
            apply_mixer_to_state(4, beta, vector), dense @ vector, atol=EXACT_TOL
        )


def test_mixer_density_matches_dense_form() -> None:
    """The fast axis-wise density mixer must equal M rho M-dagger exactly."""
    from qaoa_scheduling import apply_mixer_to_density

    rng = np.random.default_rng(3)
    n_vars = 4
    dim = 2**n_vars
    vector = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vector /= np.linalg.norm(vector)
    rho = np.outer(vector, vector.conj())

    for beta in (0.0, 0.42, 1.7):
        dense_mixer = mixer_matrix(n_vars, beta)
        expected = dense_mixer @ rho @ dense_mixer.conj().T
        actual = apply_mixer_to_density(n_vars, beta, rho)
        assert np.max(np.abs(actual - expected)) < EXACT_TOL
        # Must remain a valid density matrix.
        assert abs(np.trace(actual).real - 1.0) < EXACT_TOL
        assert np.allclose(actual, actual.conj().T, atol=EXACT_TOL)


def test_zero_angles_leave_the_uniform_superposition_alone() -> None:
    """With gamma = beta = 0 both scenarios must return the uniform distribution."""
    problem = build_problem()
    uniform = 1.0 / 2**problem.n_vars
    for probabilities in (
        probabilities_uncomputed(problem, [0.0], [0.0]),
        probabilities_naive(problem, [0.0], [0.0]),
    ):
        assert np.allclose(probabilities, uniform, atol=EXACT_TOL)


# --------------------------------------------------------------------------
# W4 / W6 -- solution quality, reported with spread
# --------------------------------------------------------------------------


def test_w4_uncomputed_qaoa_beats_random_guessing() -> None:
    """QAOA with a clean cost layer must do better than uniform guessing.

    Note what this does NOT assert: that the uncomputed scenario beats the naive
    one. That was the expected result, and at p = 1 it did not hold -- the gap
    was inside the restart-to-restart spread. Asserting it would have turned an
    honest negative finding into a failing build, so the comparison is measured
    and reported rather than baked in as a pass condition. See the README.
    """
    problem = build_problem()
    result = optimise(problem, "uncomputed", layers=1, seed=SEED, restarts=4)
    assert result.score > 0.02, (
        f"uncomputed QAOA scored {result.score:.4f}, no better than random"
    )
    assert result.expected_cost < compute_baselines(problem).random_mean


def test_w6_restart_spread_is_recorded() -> None:
    """A single optimiser run is an anecdote; the spread must be available."""
    problem = build_problem()
    result = optimise(problem, "uncomputed", layers=1, seed=SEED, restarts=4)
    assert len(result.per_restart_costs) == 4
    assert not math.isnan(result.cost_spread)
    assert result.expected_cost == min(result.per_restart_costs)


def test_optimisation_is_reproducible() -> None:
    problem = build_problem()
    first = optimise(problem, "uncomputed", layers=1, seed=SEED, restarts=3)
    second = optimise(problem, "uncomputed", layers=1, seed=SEED, restarts=3)
    assert abs(first.expected_cost - second.expected_cost) < EXACT_TOL
    assert np.allclose(first.best_params, second.best_params, atol=EXACT_TOL)


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


def _expect_value_error(callable_obj, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    try:
        callable_obj(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"{getattr(callable_obj, '__name__', callable_obj)} "
                         f"did not raise ValueError")


def test_invalid_configurations_raise_clear_errors() -> None:
    problem = build_problem()
    # The encoding only produces three-literal clauses at 3x3.
    _expect_value_error(build_problem, 4, 3)
    _expect_value_error(build_problem, 3, 5)
    # Unknown scenario name.
    _expect_value_error(optimise, problem, "magic", 1, SEED, 2)
    # Degenerate baselines cannot be normalised.
    degenerate = Baselines(
        optimum=5.0, optimum_indices=(0,), random_mean=5.0, worst=5.0,
        feasible_fraction=1.0,
    )
    _expect_value_error(normalised_score, 5.0, degenerate)


def _main() -> int:
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
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
