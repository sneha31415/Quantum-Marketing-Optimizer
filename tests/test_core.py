import itertools
import sys, pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qmo import core  # noqa: E402


def default_problem():
    return core.Problem(["Search Ads", "Influencer Posts", "Email Newsletter", "Video Pre-roll", "Local Radio"],
                        [6, 5, 5, 3, 2], [10, 8, 7, 4, 2], 10)


def reference_optimum(p):
    best = -1
    for combo in itertools.product([0, 1], repeat=p.n):
        c, r, ok = p.evaluate(combo)
        if ok:
            best = max(best, r)
    return best


def test_brute_force_matches_reference_and_is_feasible():
    p = default_problem()
    bf = core.brute_force(p)
    assert bf.feasible and bf.ret == reference_optimum(p) == 15.0
    assert core.x_to_str(bf.x) == "01100"
    assert bf.extra["n_feasible"] == 16


def test_greedy_feasible_and_suboptimal_on_default_data():
    p = default_problem()
    gr = core.greedy(p)
    assert gr.feasible and gr.ret == 14.0 < core.brute_force(p).ret


def test_random_problems_brute_force_vs_greedy_invariants():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(2, 8))
        p = core.Problem([f"c{i}" for i in range(n)], rng.integers(1, 9, n), rng.integers(0, 15, n),
                         int(rng.integers(1, 25)))
        bf, gr = core.brute_force(p), core.greedy(p)
        assert bf.feasible and gr.feasible
        assert gr.ret <= bf.ret + 1e-9 and bf.ret == reference_optimum(p)


def test_decode_is_little_endian():
    # qubit 0 is the RIGHTMOST character: '00110' -> qubits 1,2 set -> campaigns 1 and 2
    assert list(core.decode("00110", 5)) == [0, 1, 1, 0, 0]
    assert list(core.decode("10000", 5)) == [0, 0, 0, 0, 1]
    # slack qubits (higher indices) are ignored
    assert list(core.decode("1111100110"[-9:], 5)) == [0, 1, 1, 0, 0]


def test_fwht_matches_slow_definition():
    rng = np.random.default_rng(1)
    N = 4
    f = rng.normal(size=2 ** N)
    slow = np.array([sum(f[k] * (-1) ** bin(k & S).count("1") for k in range(2 ** N)) for S in range(2 ** N)])
    assert np.allclose(core.fwht(f, N), slow)


@pytest.mark.parametrize("mode", ["hinge5", "slack_qubo"])
def test_cost_model_minimum_feasible_and_hamiltonian_exact(mode):
    p = default_problem()
    bf = core.brute_force(p)
    cm = core.build_cost_model(p, mode, opt_return=bf.ret)
    k = int(np.argmin(cm.f))
    assert cm.cost_of_k[k] <= p.budget and cm.return_of_k[k] == bf.ret
    assert np.allclose(core.pauli_diagonal(cm.full_labels, cm.full_coeffs, cm.N), cm.f)
    from qiskit.quantum_info import SparsePauliOp
    H = SparsePauliOp(cm.full_labels, cm.full_coeffs)
    assert np.allclose(np.real(np.diag(H.to_matrix())), cm.f)   # cross-check with Qiskit's own matrix
    assert cm.N == (5 if mode == "hinge5" else 5 + len(core.slack_weights(10)))


def test_penalty_is_minimal_valid_and_smaller_is_invalid():
    p = default_problem()
    cm = core.build_cost_model(p, "hinge5")
    assert cm.penalty == 4.0
    # recompute with P-1: an over-budget state must win
    f3 = -cm.return_of_k + 3 * np.maximum(0, cm.cost_of_k - p.budget)
    assert cm.cost_of_k[int(np.argmin(f3))] > p.budget


def test_slack_weights_cover_exactly_zero_to_budget():
    for B in range(1, 40):
        ws = core.slack_weights(B)
        sums = {sum(c) for r in range(len(ws) + 1) for c in itertools.combinations(ws, r)}
        assert sums == set(range(B + 1))


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        core.Problem(["a", "b"], [1.5, 2], [1, 2], 3)
    with pytest.raises(ValueError):
        core.Problem(["a", "a"], [1, 2], [1, 2], 3)
    with pytest.raises(ValueError):
        core.Problem(["a", "b"], [0, 2], [1, 2], 3)
    with pytest.raises(ValueError):
        core.Problem(["a", "b"], [1, 2], [-1, 2], 3)
    with pytest.raises(ValueError):
        core.Problem(["a", "b"], [1, 2], [1, 2], 0)
    with pytest.raises(ValueError):
        core.Problem([f"c{i}" for i in range(9)], [1] * 9, [1] * 9, 3)


def test_qaoa_runs_reports_consistently_and_never_returns_infeasible():
    p = default_problem()
    bf = core.brute_force(p)
    cm = core.build_cost_model(p, "hinge5", opt_return=bf.ret)
    res = core.run_qaoa(p, cm, p=1, restarts=2, maxiter=40, shots_final=2048, seed=3)
    assert sum(res.campaign_counts.values()) == 2048
    assert abs(res.results_df["Probability"].sum() - 1) < 1e-9
    an = core.analyze_qaoa(p, res, bf)
    qa = core.qaoa_solution(p, res, an)
    assert qa is not None and qa.feasible and qa.ret <= bf.ret
    # table feasibility flags agree with independent evaluation
    for _, row in res.results_df.iterrows():
        x = [int(c) for c in row["Selection (x0..)"]]
        assert p.evaluate(x)[2] == row["Feasible"]
    assert 0 <= an.p_opt <= 1


def test_qaoa_beats_uniform_in_expectation_on_default_problem():
    """Sanity check that the optimiser actually learns something (ideal p=2 gives ~3x lift on the optimum)."""
    p = default_problem()
    bf = core.brute_force(p)
    cm = core.build_cost_model(p, "hinge5", opt_return=bf.ret)
    res = core.run_qaoa(p, cm, p=2, restarts=6, maxiter=150, shots_final=8192, seed=7)
    an = core.analyze_qaoa(p, res, bf)
    assert an.lift > 1.5
    # average energy found must be better than the uniform-superposition average
    assert res.best_energy < float(np.mean(cm.f_norm))


def test_six_campaigns_and_slack_mode_run():
    p = core.Problem(["a", "b", "c", "d", "e", "f"], [7, 6, 3, 5, 4, 2], [11, 9, 4, 8, 7, 3], 14)
    bf = core.brute_force(p)
    for mode in ("hinge5", "slack_qubo"):
        cm = core.build_cost_model(p, mode, opt_return=bf.ret)
        res = core.run_qaoa(p, cm, p=1, restarts=1, maxiter=20, shots_final=1024, seed=1)
        assert sum(res.campaign_counts.values()) == 1024


def test_too_many_qubits_is_rejected_not_silent():
    p = core.Problem([f"c{i}" for i in range(8)], [4] * 8, [5] * 8, 500)
    with pytest.raises(ValueError):
        core.build_cost_model(p, "slack_qubo")
