"""Qiskit-free part of the project: problem definition, classical solvers, result analysis.

Kept separate so the Streamlit UI thread never imports Qiskit (see worker.py for why).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

MAX_CAMPAIGNS = 8
MAX_QUBITS = 12


@dataclass
class Problem:
    names: list
    costs: np.ndarray      # positive integers
    returns: np.ndarray    # non-negative numbers
    budget: int

    def __post_init__(self):
        self.costs = np.asarray(self.costs)
        self.returns = np.asarray(self.returns, dtype=float)
        self.validate()
        self.costs = self.costs.astype(int)
        self.budget = int(self.budget)

    @property
    def n(self) -> int:
        return len(self.names)

    def validate(self):
        n = len(self.names)
        if not (2 <= n <= MAX_CAMPAIGNS):
            raise ValueError(f"Number of campaigns must be between 2 and {MAX_CAMPAIGNS}.")
        if len(set(self.names)) != n or any(not str(s).strip() for s in self.names):
            raise ValueError("Campaign names must be non-empty and unique.")
        if len(self.costs) != n or len(self.returns) != n:
            raise ValueError("Names, costs and returns must have the same length.")
        if not np.all(np.isfinite(self.costs.astype(float))) or np.any(self.costs <= 0) \
                or np.any(self.costs != np.round(self.costs)):
            raise ValueError("Costs must be positive integers.")
        if not np.all(np.isfinite(self.returns)) or np.any(self.returns < 0):
            raise ValueError("Returns must be non-negative numbers.")
        if self.budget <= 0 or self.budget != int(self.budget):
            raise ValueError("Budget must be a positive integer.")

    def evaluate(self, x):
        x = np.asarray(x, dtype=int)
        cost = int(self.costs @ x)
        return cost, float(self.returns @ x), cost <= self.budget

    def selected(self, x):
        return [self.names[i] for i in range(self.n) if x[i] == 1]


@dataclass
class Solution:
    label: str
    x: np.ndarray
    cost: int
    ret: float
    feasible: bool
    seconds: float
    extra: dict = field(default_factory=dict)


def make_solution(problem: Problem, label: str, x, seconds: float, **extra) -> Solution:
    c, r, ok = problem.evaluate(x)
    return Solution(label, np.asarray(x, dtype=int), c, r, ok, seconds, extra)


def x_to_str(x) -> str:
    return "".join(str(int(v)) for v in x)


def decode(bitstring: str, n: int) -> np.ndarray:
    """Decode a Qiskit measurement string into campaign vector x (little-endian aware)."""
    bits = [int(b) for b in bitstring[::-1]]
    return np.array(bits[:n], dtype=int)


# ----------------------------------------------------------------------------- classical
def brute_force(problem: Problem) -> Solution:
    t0 = time.perf_counter()
    best_x, best_r, n_feas = None, -np.inf, 0
    for combo in itertools.product([0, 1], repeat=problem.n):
        c, r, ok = problem.evaluate(combo)
        if ok:
            n_feas += 1
            if r > best_r:
                best_r, best_x = r, np.array(combo)
    return make_solution(problem, "Brute force (exact)", best_x, time.perf_counter() - t0,
                         n_feasible=n_feas, n_total=2 ** problem.n)


def greedy(problem: Problem) -> Solution:
    t0 = time.perf_counter()
    order = np.argsort(-(problem.returns / problem.costs), kind="stable")
    x, spent = np.zeros(problem.n, dtype=int), 0
    for i in order:
        if spent + problem.costs[i] <= problem.budget:
            x[i] = 1
            spent += problem.costs[i]
    return make_solution(problem, "Greedy (return / cost)", x, time.perf_counter() - t0,
                         order=[problem.names[i] for i in order])


def slack_weights(budget: int) -> list:
    """Weights whose subset sums cover exactly 0..budget (powers of two + remainder)."""
    ws, p, rem = [], 1, int(budget)
    while rem >= p:
        ws.append(p)
        rem -= p
        p *= 2
    if rem > 0:
        ws.append(rem)
    return ws


@dataclass
class CostModel:
    """Plain-data description of the QAOA cost function (no Qiskit objects)."""
    mode: str
    n: int
    N: int
    slack_w: list
    penalty: float
    f: np.ndarray                 # cost of every basis state (original units)
    scale: float
    full_labels: list             # exact cost Hamiltonian, Pauli labels (incl. identity)
    full_coeffs: np.ndarray
    evolve_labels: list           # normalised Hamiltonian used in the circuit (identity removed)
    evolve_coeffs: np.ndarray
    penalty_table: pd.DataFrame
    max_pauli_weight: int
    cost_of_k: np.ndarray
    return_of_k: np.ndarray

    @property
    def f_norm(self):
        return self.f / self.scale


@dataclass
class QAOAResult:
    p: int
    theta: np.ndarray
    best_energy: float
    history: list
    results_df: pd.DataFrame        # campaign-level measurement table
    campaign_counts: dict
    shots: int
    opt_seconds: float
    total_seconds: float
    transpiled_depth: int
    gate_counts: dict
    n_circuit_evals: int


def measurement_table(problem: Problem, campaign_counts: dict, shots: int) -> pd.DataFrame:
    rows = []
    for key in (x_to_str(c) for c in itertools.product([0, 1], repeat=problem.n)):
        x = np.array([int(ch) for ch in key])
        c, r, ok = problem.evaluate(x)
        cnt = campaign_counts.get(key, 0)
        rows.append({"Selection (x0..)": key, "Qiskit string": key[::-1],
                     "Selected": ", ".join(problem.selected(x)) or "(none)", "Cost": c, "Return": r,
                     "Feasible": ok, "Counts": cnt, "Probability": cnt / shots})
    return pd.DataFrame(rows).sort_values(["Counts", "Return"], ascending=False).reset_index(drop=True)


@dataclass
class QAOAAnalysis:
    top: pd.Series
    best_feasible: Optional[pd.Series]
    p_opt: float
    lift: float
    feasible_share: float
    cond_return: Optional[float]      # mean return of feasible shots
    approx_ratio: Optional[float]     # cond_return / optimum


def analyze_qaoa(problem: Problem, res: QAOAResult, bf: Solution) -> QAOAAnalysis:
    df = res.results_df
    top = df.iloc[0]
    feas = df[df["Feasible"] & (df["Counts"] > 0)]
    best_feas = feas.sort_values(["Return", "Counts"], ascending=False).iloc[0] if len(feas) else None
    p_opt = float(res.campaign_counts.get(x_to_str(bf.x), 0) / res.shots)
    feas_share = float(df.loc[df.Feasible, "Probability"].sum())
    cond = None
    if len(feas):
        cond = float((feas["Return"] * feas["Counts"]).sum() / feas["Counts"].sum())
    ratio = (cond / bf.ret) if (cond is not None and bf.ret > 0) else None
    return QAOAAnalysis(top, best_feas, p_opt, p_opt * 2 ** problem.n, feas_share, cond, ratio)


def qaoa_solution(problem: Problem, res: QAOAResult, an: QAOAAnalysis) -> Optional[Solution]:
    """Best FEASIBLE measured bitstring (None if QAOA never produced a feasible sample)."""
    if an.best_feasible is None:
        return None
    x = np.array([int(ch) for ch in an.best_feasible["Selection (x0..)"]])
    sol = make_solution(problem, "QAOA (best feasible sample)", x, res.total_seconds)
    assert sol.feasible
    return sol


def random_baseline(problem: Problem, shots: int, seed: int = 99) -> dict:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, 2, size=(shots, problem.n))
    ok = [problem.evaluate(v)[1] for v in samples if problem.evaluate(v)[2]]
    return {"feasible_share": len(ok) / shots, "best_return": max(ok) if ok else None}


def comparison_table(problem: Problem, bf: Solution, gr: Solution, qa: Optional[Solution],
                     top_row: Optional[pd.Series]) -> pd.DataFrame:
    rows = [bf, gr]
    if qa is not None:
        rows.append(qa)
    out = []
    for s in rows:
        out.append({"Method": s.label, "Selected": ", ".join(problem.selected(s.x)) or "(none)",
                    "Total cost": s.cost, "Expected return": s.ret,
                    "Within budget?": "Yes" if s.feasible else "NO", "Time (s)": s.seconds,
                    "Gap to optimum": bf.ret - s.ret})
    if qa is None:
        out.append({"Method": "QAOA (best feasible sample)", "Selected": "NO FEASIBLE SAMPLE", "Total cost": None,
                    "Expected return": None, "Within budget?": "-", "Time (s)": None, "Gap to optimum": None})
    if top_row is not None:
        out.append({"Method": "QAOA (most frequent, unfiltered)", "Selected": top_row["Selected"],
                    "Total cost": int(top_row["Cost"]), "Expected return": float(top_row["Return"]),
                    "Within budget?": "Yes" if top_row["Feasible"] else "NO (rejected)",
                    "Time (s)": None,
                    "Gap to optimum": (bf.ret - float(top_row["Return"])) if top_row["Feasible"] else None})
    return pd.DataFrame(out)
