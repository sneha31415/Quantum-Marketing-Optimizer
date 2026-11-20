"""Qiskit part of the project: cost Hamiltonian, QAOA circuit, simulation on Qiskit Aer.

Bit-ordering convention (IMPORTANT)
-----------------------------------
Campaign i <-> qubit i. Qiskit prints qubit 0 as the RIGHTMOST character of a measured string, so
`decode()` reverses the string. Integer k of a basis state has bit i = value of qubit i.

NOTE: this module imports Qiskit. Inside the Streamlit app it must only be imported by the persistent worker
thread (qmo/worker.py). See the docstring there.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

# Re-export the Qiskit-free API so `from qmo import core` gives access to everything.
from .model import *  # noqa: F401,F403
from .model import (MAX_QUBITS, CostModel, Problem, QAOAResult, Solution, analyze_qaoa, brute_force,  # noqa: F401
                    decode, measurement_table, slack_weights, x_to_str)


def fwht(f, N: int) -> np.ndarray:
    """Fast Walsh-Hadamard transform: out[S] = sum_k f[k] (-1)^{popcount(k & S)}."""
    a = np.array(f, dtype=float).reshape([2] * N)
    for ax in range(N):
        a0, a1 = np.take(a, 0, axis=ax), np.take(a, 1, axis=ax)
        a = np.stack([a0 + a1, a0 - a1], axis=ax)
    return a.reshape(-1)


def pauli_label(S: int, N: int) -> str:
    return "".join("Z" if (S >> i) & 1 else "I" for i in reversed(range(N)))


def pauli_diagonal(labels, coeffs, N: int) -> np.ndarray:
    """Diagonal of a Z-only Pauli sum computed directly from its labels (independent of the Walsh-Hadamard code and
    of Qiskit's matrix builder). Label char at string position j acts on qubit N-1-j (little-endian)."""
    K = np.arange(2 ** N)
    diag = np.zeros(2 ** N)
    for lab, c in zip(labels, np.real(coeffs)):
        mask = 0
        for j, ch in enumerate(lab):
            if ch == "Z":
                mask |= 1 << (N - 1 - j)
            elif ch != "I":
                raise ValueError("Only I and Z are expected in a diagonal cost Hamiltonian")
        parity = np.zeros(2 ** N, dtype=int)
        q = 0
        while mask >> q:
            if (mask >> q) & 1:
                parity ^= (K >> q) & 1
            q += 1
        diag += c * (1 - 2 * parity)
    return diag


# ----------------------------------------------------------------------------- cost model
def build_cost_model(problem: Problem, mode: str = "hinge5", opt_return: Optional[float] = None) -> CostModel:
    if mode not in ("hinge5", "slack_qubo"):
        raise ValueError("mode must be 'hinge5' or 'slack_qubo'")
    n = problem.n
    slack_w = [] if mode == "hinge5" else slack_weights(problem.budget)
    N = n + len(slack_w)
    if N > MAX_QUBITS:
        raise ValueError(f"This setup needs {N} qubits (limit {MAX_QUBITS}). Use hinge mode or a smaller budget.")
    if opt_return is None:
        opt_return = brute_force(problem).ret

    K = np.arange(2 ** N)
    bits = ((K[:, None] >> np.arange(N)[None, :]) & 1).astype(int)
    X, Y = bits[:, :n], bits[:, n:]
    cost_of_k, return_of_k = X @ problem.costs, X @ problem.returns
    B = problem.budget

    def cost_function(P):
        if mode == "hinge5":
            return -return_of_k + P * np.maximum(0, cost_of_k - B)
        return -return_of_k + P * (cost_of_k + Y @ np.array(slack_w, dtype=int) - B) ** 2

    def valid(P):
        ff = cost_function(P)
        best = np.flatnonzero(np.isclose(ff, ff.min()))
        return bool(all(cost_of_k[k] <= B and np.isclose(return_of_k[k], opt_return) for k in best))

    penalty, table = None, []
    for P in range(1, int(np.ceil(problem.returns.max())) + 2):
        ok = valid(float(P))
        table.append({"Penalty P": P, "Optimum stays feasible?": "Yes" if ok else "No (over-budget state wins)"})
        if ok:
            penalty = float(P)
            break
    if penalty is None:          # cannot happen for integer costs (proof in notebook), but never fail silently
        raise RuntimeError("No valid penalty found; costs must be integers.")
    f = cost_function(penalty)

    coeffs = fwht(f, N) / 2 ** N
    keep = [S for S in range(2 ** N) if abs(coeffs[S]) > 1e-9]
    full_labels = [pauli_label(S, N) for S in keep]
    full_coeffs = np.array([coeffs[S] for S in keep])
    # verification: independent diagonal from labels must equal the cost function exactly
    if not np.allclose(pauli_diagonal(full_labels, full_coeffs, N), f):
        raise RuntimeError("Hamiltonian does not reproduce the cost function.")

    scale = float(f.max() - f.min()) or 1.0
    id_label = "I" * N
    terms = [(lab, c / scale) for lab, c in zip(full_labels, full_coeffs) if lab != id_label]
    if not terms:                # degenerate flat cost function
        terms = [("I" * (N - 1) + "Z", 1e-12)]
    max_w = max(lab.count("Z") for lab in full_labels)
    return CostModel(mode, n, N, slack_w, penalty, f, scale, full_labels, full_coeffs,
                     [t[0] for t in terms], np.array([t[1] for t in terms]),
                     pd.DataFrame(table), max_w, cost_of_k, return_of_k)


# ----------------------------------------------------------------------------- QAOA
def build_qaoa_circuit(cm: CostModel, p: int, measure: bool = True):
    H_evolve = SparsePauliOp(cm.evolve_labels, cm.evolve_coeffs)
    gammas, betas = ParameterVector("γ", p), ParameterVector("β", p)
    qc = QuantumCircuit(cm.N, name="QAOA")
    qc.h(range(cm.N))
    for layer in range(p):
        qc.barrier()
        qc.append(PauliEvolutionGate(H_evolve, time=gammas[layer], label=f"Cost layer {layer + 1}"), range(cm.N))
        qc.barrier()
        for q in range(cm.N):
            qc.rx(2 * betas[layer], q)
    if measure:
        qc.measure_all()
    return qc, gammas, betas


def run_qaoa(problem: Problem, cm: CostModel, p: int = 2, restarts: int = 6, maxiter: int = 150,
             shots_opt: int = 1024, shots_final: int = 8192, seed: int = 7,
             progress: Optional[Callable[[int, int, float], None]] = None) -> QAOAResult:
    backend = AerSimulator()
    qc, gam, bet = build_qaoa_circuit(cm, p)
    qt = transpile(qc, backend, optimization_level=1, seed_transpiler=seed)   # transpile once, rebind many times
    params = list(gam) + list(bet)
    f_norm = cm.f_norm
    rng = np.random.default_rng(seed)

    def energy(theta, shots):
        bound = qt.assign_parameters(dict(zip(params, theta)))
        counts = backend.run(bound, shots=shots, seed_simulator=seed).result().get_counts()
        return sum(c * f_norm[int(b, 2)] for b, c in counts.items()) / shots

    t0 = time.perf_counter()
    best, history, n_eval = None, [], 0
    for r in range(restarts):
        theta0 = np.concatenate([rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)])
        trace = []

        def obj(th):
            e = energy(th, shots_opt)
            trace.append(e)
            return e

        res = minimize(obj, theta0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
        history.append(trace)
        n_eval += len(trace)
        if best is None or res.fun < best.fun:
            best = res
        if progress:
            progress(r + 1, restarts, float(res.fun))
    opt_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    bound = qt.assign_parameters(dict(zip(params, best.x)))
    counts = backend.run(bound, shots=shots_final, seed_simulator=seed + 1).result().get_counts()
    total = opt_seconds + (time.perf_counter() - t1)

    n = problem.n
    campaign_counts = {}
    for bs, c in counts.items():
        key = x_to_str(decode(bs, n))
        campaign_counts[key] = campaign_counts.get(key, 0) + c
    df = measurement_table(problem, campaign_counts, shots_final)
    return QAOAResult(p, np.array(best.x), float(best.fun), history, df, campaign_counts, shots_final, opt_seconds,
                      total, qt.depth(), dict(qt.count_ops()), n_eval)


def depth_study(problem: Problem, cm: CostModel, bf: Solution, depths=(1, 2, 3), restarts: int = 4,
                maxiter: int = 100, shots_final: int = 8192, seed: int = 7,
                progress: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(depths):
        res = run_qaoa(problem, cm, p=p, restarts=restarts, maxiter=maxiter, shots_final=shots_final, seed=seed)
        an = analyze_qaoa(problem, res, bf)
        rows.append({"Layers p": p, "P(optimum)": an.p_opt, "Lift vs random": an.lift,
                     "Feasible share": an.feasible_share, "Transpiled depth": res.transpiled_depth,
                     "Time (s)": res.total_seconds})
        if progress:
            progress(i + 1, len(depths))
    return pd.DataFrame(rows)


def draw_circuit(cm: CostModel, res: QAOAResult):
    """Matplotlib figure of the QAOA circuit with the optimised angles substituted in."""
    qc, g, b = build_qaoa_circuit(cm, res.p, measure=True)
    qc = qc.assign_parameters(dict(zip(list(g) + list(b), res.theta)))
    return qc.draw("mpl", fold=-1, scale=0.8, style="iqp")
