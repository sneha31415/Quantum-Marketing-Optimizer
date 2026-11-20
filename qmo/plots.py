"""Matplotlib figures used by the Streamlit app (and usable from notebooks)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from .model import Problem, QAOAResult

GREEN, RED, GOLD, PURPLE, TEAL = "#2a9d8f", "#e76f51", "#e9c46a", "#7b5ea7", "#2a9d8f"


def fig_cost_return(problem: Problem):
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    sc = ax.scatter(problem.costs, problem.returns, s=190, c=problem.returns / problem.costs,
                    cmap="viridis", edgecolor="k", zorder=3)
    for i, nm in enumerate(problem.names):
        ax.annotate(nm, (problem.costs[i], problem.returns[i]), textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.axvline(problem.budget, color="red", ls="--", label=f"Budget = {problem.budget}")
    ax.set_xlabel("Cost"); ax.set_ylabel("Expected return")
    ax.set_title("Campaign cost vs expected return"); ax.grid(alpha=0.3); ax.legend()
    fig.colorbar(sc, ax=ax, label="Return per unit cost")
    fig.tight_layout()
    return fig


def fig_selection(problem: Problem, rows: dict):
    """rows: {label: x-vector}"""
    labels = list(rows)
    M = np.array([rows[k] for k in labels])
    fig, ax = plt.subplots(figsize=(7.5, 0.9 + 0.7 * len(labels)))
    ax.imshow(M, cmap="Greens", vmin=0, vmax=1.4, aspect="auto")
    ax.set_xticks(range(problem.n)); ax.set_xticklabels(problem.names, rotation=20, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(problem.n):
            ax.text(j, i, "Selected" if M[i, j] else "-", ha="center", va="center",
                    color="white" if M[i, j] else "gray", fontsize=9, fontweight="bold")
    ax.set_title("Campaign selection by algorithm")
    fig.tight_layout()
    return fig


def fig_objective(labels, values, optimum):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    cols = [TEAL, GOLD, PURPLE, "#999999"][: len(labels)]
    bars = ax.bar(labels, values, color=cols, edgecolor="k")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontweight="bold")
    ax.axhline(optimum, color="k", ls=":", label="Exact optimum")
    ax.set_ylabel("Total expected return (feasible solutions only)")
    ax.set_title("Objective value: classical vs quantum")
    top = max(values + [optimum])
    ax.legend(); ax.set_ylim(0, top * 1.2 if top > 0 else 1.0)      # avoid a degenerate axis when every return is 0
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    return fig


def fig_distribution(problem: Problem, res: QAOAResult, opt_key: str):
    df = res.results_df.sort_values("Probability", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(9, 0.42 * len(df)), 4.6))
    cols = [GREEN if ok else RED for ok in df["Feasible"]]
    edge = ["k" if s == opt_key else "none" for s in df["Selection (x0..)"]]
    ax.bar(range(len(df)), df["Probability"], color=cols, edgecolor=edge, linewidth=2)
    ax.set_xticks(range(len(df))); ax.set_xticklabels(df["Selection (x0..)"], rotation=90, fontsize=7)
    ax.axhline(1 / 2 ** problem.n, color="gray", ls="--")
    ax.legend(handles=[Patch(color=GREEN, label="Feasible"), Patch(color=RED, label="Infeasible (over budget)"),
                       Patch(facecolor="white", edgecolor="k", linewidth=2, label="True optimum"),
                       Line2D([0], [0], color="gray", ls="--", label="Uniform random")])
    ax.set_xlabel("Campaign selection string (digit i = campaign i; 1 = selected)")
    ax.set_ylabel("Measured probability")
    ax.set_title(f"QAOA measurement distribution (p={res.p}, {res.shots} shots, Qiskit Aer)")
    fig.tight_layout()
    return fig


def fig_convergence(res: QAOAResult, floor: float):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, tr in enumerate(res.history):
        ax.plot(tr, alpha=0.85, label=f"restart {i + 1}")
    ax.axhline(floor, color="k", ls=":", label="Lowest possible")
    ax.set_xlabel("Circuit evaluation"); ax.set_ylabel("Normalised expected cost")
    ax.set_title("COBYLA convergence"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def fig_depth(df):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(df["Layers p"].astype(str), df["P(optimum)"], color=PURPLE, edgecolor="k")
    ax.set_xlabel("QAOA layers p"); ax.set_ylabel("Probability of true optimum")
    ax.set_title("Effect of circuit depth")
    for i, v in enumerate(df["P(optimum)"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    return fig
