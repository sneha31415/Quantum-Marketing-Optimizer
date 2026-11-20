"""Quantum Marketing Optimizer - Streamlit UI.

Run with:  streamlit run app.py
"""
import time

import numpy as np
import pandas as pd
import streamlit as st

# NOTE: only the Qiskit-free modules are imported here. All Qiskit work runs in one persistent worker thread
# (see qmo/worker.py for the reason: Qiskit 2.x can segfault if used from a different thread than the one that imported it).
from qmo import model, plots
from qmo.worker import QuantumWorker


@st.cache_resource
def get_worker() -> QuantumWorker:
    return QuantumWorker()


worker = get_worker()

st.set_page_config(page_title="Quantum Marketing Optimizer", page_icon="⚛️", layout="wide")

# ----------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(120deg,#3a1c71 0%,#6a3fa0 45%,#2a9d8f 100%); color: white;
       padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: .3rem 0 0 0; opacity: .92; font-size: 1.02rem;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .6rem;}
.ok {color: #2a9d8f; font-weight: 700;} .bad {color: #e76f51; font-weight: 700;}
div[data-testid="stMetric"] {background: rgba(106,63,160,.07); border: 1px solid rgba(106,63,160,.25);
       padding: .6rem .9rem; border-radius: 10px;}
</style>
<div class="hero">
  <h1>⚛️ Quantum Marketing Optimizer</h1>
  <p>Selecting advertising campaigns under a budget with <b>QAOA</b> (Qiskit + Qiskit Aer), compared against exact brute force and a greedy heuristic.</p>
</div>
""",
    unsafe_allow_html=True,
)

PRESETS = {
    "Default (greedy misses the optimum)": (
        ["Search Ads", "Influencer Posts", "Email Newsletter", "Video Pre-roll", "Local Radio"],
        [6, 5, 5, 3, 2], [10, 8, 7, 4, 2], 10),
    "Tight budget": (
        ["Search Ads", "Influencer Posts", "Email Newsletter", "Video Pre-roll", "Local Radio"],
        [6, 5, 5, 3, 2], [10, 8, 7, 4, 2], 7),
    "6 campaigns": (
        ["TV Spot", "Billboards", "Podcast Ads", "Search Ads", "Social Media", "Flyers"],
        [7, 6, 3, 5, 4, 2], [11, 9, 4, 8, 7, 3], 14),
}

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Problem")
    preset = st.selectbox("Preset dataset", list(PRESETS) + ["Custom (edit below)"])
    if preset in PRESETS:
        nm, co, re_, bd = PRESETS[preset]
        base = pd.DataFrame({"Campaign": nm, "Cost": co, "Return": re_})
    else:
        base = pd.DataFrame({"Campaign": ["A", "B", "C", "D", "E"], "Cost": [4, 3, 5, 2, 6],
                             "Return": [7, 4, 8, 3, 9]})
        bd = 10
    st.caption("Edit the table (costs must be positive integers). Add or remove rows with the table controls.")
    table = st.data_editor(base, num_rows="dynamic", width="stretch", key=f"tbl_{preset}",
                           column_config={"Cost": st.column_config.NumberColumn(min_value=1, step=1),
                                          "Return": st.column_config.NumberColumn(min_value=0.0, step=0.5)})
    budget = st.number_input("Budget", min_value=1, max_value=500, value=int(bd), step=1, key=f"bud_{preset}")

    st.header("2. QAOA settings")
    mode_label = st.radio("Budget encoding",
                          ["5-qubit hinge penalty (1 qubit per campaign)", "Slack-variable QUBO (extra qubits)"])
    mode = "hinge5" if mode_label.startswith("5") else "slack_qubo"
    p_layers = st.slider("Layers p", 1, 3, 2)
    restarts = st.slider("Optimizer restarts", 1, 10, 6)
    maxiter = st.slider("Max COBYLA iterations", 30, 300, 150, step=10)
    shots_final = st.select_slider("Final shots", options=[1024, 2048, 4096, 8192, 16384], value=8192)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=7, step=1)
    do_depth = st.checkbox("Also run depth study (p = 1, 2, 3)", value=False,
                           help="Runs QAOA three more times to show how circuit depth changes the result.")
    run = st.button("▶ Run optimization", type="primary", width="stretch")
    v = worker.versions()
    st.caption(f"Qiskit {v['qiskit']} · Aer {v['qiskit-aer']}")


# ----------------------------------------------------------------------------- build problem
def parse_problem():
    df = table.dropna(how="any")
    if len(df) != len(table):
        raise ValueError("Please fill in every cell of the table (or delete incomplete rows).")
    names = [str(s).strip() for s in df["Campaign"]]
    costs = df["Cost"].to_numpy(dtype=float)
    if np.any(costs != np.round(costs)):
        raise ValueError("Costs must be whole numbers.")
    return model.Problem(names, costs.astype(int), df["Return"].to_numpy(dtype=float), int(budget))


try:
    problem = parse_problem()
    problem_error = None
except Exception as e:  # shown to the user, app keeps running
    problem, problem_error = None, str(e)

if problem_error:
    st.error(f"Invalid input: {problem_error}")
    st.stop()

# ----------------------------------------------------------------------------- run
def wait_for(fut, bar, state, label):
    """Poll a worker job from the UI thread, updating a progress bar."""
    while not fut.done():
        frac = state["i"] / max(state["total"], 1)
        bar.progress(min(frac, 0.99), text=f"{label}: step {state['i']}/{state['total']}")
        time.sleep(0.15)
    return fut.result()


if run:
    try:
        bf = model.brute_force(problem)
        gr = model.greedy(problem)
        cm = worker.call("build_cost_model", problem, mode, opt_return=bf.ret)
        state = {"i": 0, "total": restarts}
        bar = st.progress(0.0, text="Running QAOA on Qiskit Aer...")

        def cb(i, total, e):
            state["i"], state["total"] = i, total

        fut = worker.submit("run_qaoa", problem, cm, p=p_layers, restarts=restarts, maxiter=maxiter,
                            shots_final=shots_final, seed=int(seed), progress=cb)
        res = wait_for(fut, bar, state, "QAOA restart")
        bar.empty()
        an = model.analyze_qaoa(problem, res, bf)
        qa = model.qaoa_solution(problem, res, an)
        base_rand = model.random_baseline(problem, shots_final)
        circuit_fig = worker.call("draw_circuit", cm, res)
        depth_df = None
        if do_depth:
            dstate = {"i": 0, "total": 3}
            dbar = st.progress(0.0, text="Depth study...")

            def dcb(i, t):
                dstate["i"], dstate["total"] = i, t

            dfut = worker.submit("depth_study", problem, cm, bf, shots_final=shots_final, seed=int(seed), progress=dcb)
            depth_df = wait_for(dfut, dbar, dstate, "Depth study (p = 1, 2, 3)")
            dbar.empty()
        st.session_state["out"] = dict(problem=problem, bf=bf, gr=gr, cm=cm, res=res, an=an, qa=qa,
                                       rand=base_rand, depth=depth_df, mode=mode, circuit_fig=circuit_fig)
    except Exception as e:
        st.session_state.pop("out", None)
        st.error(f"Could not run: {e}")

out = st.session_state.get("out")
tabs = st.tabs(["📋 Problem", "🧮 Formulation", "🏁 Results", "📊 Charts", "🔬 QAOA details", "🧠 Circuit", "📝 About"])

# ----------------------------------------------------------------------------- tab: problem
with tabs[0]:
    c1, c2 = st.columns([1, 1.15])
    with c1:
        show = pd.DataFrame({"Campaign": problem.names, "Cost": problem.costs, "Return": problem.returns,
                             "Return per cost": np.round(problem.returns / problem.costs, 3)})
        st.dataframe(show, hide_index=True, width="stretch")
        st.markdown(f"**Budget:** {problem.budget}  \n**Total cost of all campaigns:** {int(problem.costs.sum())}  \n"
                    f"**Possible selections:** 2^{problem.n} = {2 ** problem.n}")
        if not out:
            st.info("Press **▶ Run optimization** in the sidebar to solve it.")
    with c2:
        st.pyplot(plots.fig_cost_return(problem))

# ----------------------------------------------------------------------------- tab: formulation
with tabs[1]:
    st.markdown(r"""
**Binary variables:** $x_i = 1$ if campaign $i$ is selected, else $0$.

**Objective:** maximise $\sum_i r_i x_i$ subject to $\sum_i c_i x_i \le B$  (a 0/1 knapsack problem).

**Why not $P(\sum c_i x_i - B)^2$ directly?** It penalises *any* deviation from $B$, so it wrongly punishes
under-spending (it models an equality, not $\le$).
""")
    a, b = st.columns(2)
    with a:
        st.markdown("##### Mode A: 5-qubit hinge penalty (default)")
        st.latex(r"f(x) = -\sum_i r_i x_i + P\,\max\!\left(0,\ \sum_i c_i x_i - B\right)")
        st.caption("Exactly one qubit per campaign. The max(0, ·) term is not quadratic, so its Z-expansion has "
                   "higher-order terms (a HUBO, not a strict QUBO).")
    with b:
        st.markdown("##### Mode B: slack-variable QUBO")
        st.latex(r"f(x,y) = -\sum_i r_i x_i + P\Big(\sum_i c_i x_i + \sum_k w_k y_k - B\Big)^2")
        st.caption("A true QUBO, but the slack bits y_k add extra qubits (about log₂B + 1).")
    st.markdown(r"""
**Penalty strength.** Too small and an over-budget bitstring beats the best feasible one; too large and the
return signal is drowned out. The app tests $P = 1, 2, 3, \dots$ and picks the **smallest integer that is verified
exhaustively** to keep the global minimum feasible and equal to the brute-force optimum.
(Safe upper bound: $P = 1 + \max_i r_i$, provably sufficient for integer costs.)

**Cost Hamiltonian.** With $x_i = (1 - Z_i)/2$, the cost function becomes a diagonal Pauli-$Z$ Hamiltonian
$H_C = \sum_S a_S \prod_{i \in S} Z_i$ via a Walsh-Hadamard transform. The app checks that $H_C$ reproduces $f$ exactly.

**Bit order.** Qiskit prints qubit 0 as the *rightmost* character, so all measured strings are reversed before
decoding into campaigns.
""")
    if out:
        cm = out["cm"]
        st.markdown("##### Verified values for this run")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Qubits", cm.N, help="Campaign qubits + slack qubits")
        m2.metric("Slack qubits", len(cm.slack_w))
        m3.metric("Chosen penalty P", f"{cm.penalty:g}")
        m4.metric("Max Pauli weight", cm.max_pauli_weight)
        st.dataframe(cm.penalty_table, hide_index=True)
        st.success("Verified: the cost function's minimum is feasible and equals the brute-force optimum; "
                   "the Hamiltonian's diagonal equals the cost function exactly.")

# ----------------------------------------------------------------------------- tab: results
with tabs[2]:
    if not out:
        st.info("Run the optimization first (sidebar).")
    else:
        P_, bf, gr, qa, an, res = out["problem"], out["bf"], out["gr"], out["qa"], out["an"], out["res"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Optimal return (brute force)", f"{bf.ret:g}")
        k2.metric("Greedy return", f"{gr.ret:g}", delta=f"{gr.ret - bf.ret:g}" if gr.ret != bf.ret else "optimal",
                  delta_color="off" if gr.ret == bf.ret else "inverse")
        if qa is not None:
            k3.metric("QAOA best feasible return", f"{qa.ret:g}",
                      delta=f"{qa.ret - bf.ret:g}" if qa.ret != bf.ret else "optimal",
                      delta_color="off" if qa.ret == bf.ret else "inverse")
        else:
            k3.metric("QAOA best feasible return", "none found")
        k4.metric("P(optimum) under QAOA", f"{an.p_opt:.1%}", delta=f"{an.lift:.2f}× random", delta_color="off")

        st.markdown("#### Comparison table")
        comp = model.comparison_table(P_, bf, gr, qa, an.top)
        st.dataframe(comp.style.format({"Expected return": "{:.1f}", "Time (s)": "{:.4f}", "Gap to optimum": "{:.1f}"},
                                       na_rep="-"), hide_index=True, width="stretch")
        st.caption("QAOA time = classical optimisation loop + all simulated circuit runs on Aer (a classical CPU). "
                   "It is not comparable to the microsecond classical times, and is not quantum hardware time.")

        if not an.top["Feasible"]:
            st.warning(f"QAOA's single most frequent bitstring `{an.top['Selection (x0..)']}` "
                       f"({an.top['Selected']}, cost {int(an.top['Cost'])} > budget {P_.budget}) is **over budget** and "
                       "was rejected. The reported QAOA solution is the best *feasible* measured bitstring.")
        else:
            st.success(f"QAOA's most frequent bitstring `{an.top['Selection (x0..)']}` is feasible.")

        if qa is None:
            st.error("QAOA produced no feasible sample in this run.")
        elif np.isclose(qa.ret, bf.ret):
            st.info("QAOA's best feasible sample matches the exact optimum in this run.")
        else:
            st.warning(f"QAOA's best feasible sample (return {qa.ret:g}) did **not** reach the optimum ({bf.ret:g}). "
                       "QAOA is a heuristic and is not guaranteed to find the optimum.")
        rand = out["rand"]
        st.markdown(
            f"**Reality check.** With only {2 ** P_.n} possible outcomes and {res.shots} shots, almost any sampler "
            f"sees every bitstring. Uniform random sampling also reaches best feasible return "
            f"**{rand['best_return']:g}**. So the meaningful QAOA metrics are its probability on the optimum "
            f"(**{an.p_opt:.1%}** vs **{1 / 2 ** P_.n:.1%}** for random) and its feasible share "
            f"(**{an.feasible_share:.1%}** vs **{rand['feasible_share']:.1%}** random).")

# ----------------------------------------------------------------------------- tab: charts
with tabs[3]:
    if not out:
        st.info("Run the optimization first (sidebar).")
    else:
        P_, bf, gr, qa, an, res = out["problem"], out["bf"], out["gr"], out["qa"], out["an"], out["res"]
        c1, c2 = st.columns(2)
        with c1:
            rows = {"Brute force": bf.x, "Greedy": gr.x}
            if qa is not None:
                rows["QAOA (best feasible)"] = qa.x
            st.pyplot(plots.fig_selection(P_, rows))
        with c2:
            labels, vals = ["Brute force", "Greedy"], [bf.ret, gr.ret]
            if qa is not None:
                labels.append("QAOA")
                vals.append(qa.ret)
            st.pyplot(plots.fig_objective(labels, vals, bf.ret))
        st.pyplot(plots.fig_distribution(P_, res, model.x_to_str(bf.x)))
        if out["depth"] is not None:
            st.markdown("#### Depth study")
            d1, d2 = st.columns([1, 1.2])
            d1.dataframe(out["depth"].style.format({"P(optimum)": "{:.3f}", "Lift vs random": "{:.2f}",
                                                    "Feasible share": "{:.3f}", "Time (s)": "{:.2f}"}),
                         hide_index=True)
            d2.pyplot(plots.fig_depth(out["depth"]))
            st.caption("More layers can raise the probability of the optimum, but the circuit gets deeper "
                       "(more noise-sensitive on real hardware) and the optimisation harder.")

# ----------------------------------------------------------------------------- tab: QAOA details
with tabs[4]:
    if not out:
        st.info("Run the optimization first (sidebar).")
    else:
        res, an, P_ = out["res"], out["an"], out["problem"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Circuit evaluations", res.n_circuit_evals)
        s2.metric("Transpiled depth", res.transpiled_depth)
        s3.metric("Optimiser time (s)", f"{res.opt_seconds:.2f}")
        s4.metric("Final shots", res.shots)
        g = np.round(res.theta[:res.p], 3)
        b = np.round(res.theta[res.p:], 3)
        st.markdown(f"**Optimal angles:** γ = {list(g)}, β = {list(b)}  |  **Gate counts:** `{res.gate_counts}`")
        st.pyplot(plots.fig_convergence(res, out["cm"].f_norm.min()))
        st.markdown("#### Measured bitstrings (all outcomes, sorted by count)")
        st.dataframe(res.results_df.style.format({"Probability": "{:.4f}", "Return": "{:.1f}"}),
                     hide_index=True, width="stretch")
        st.caption("'Selection' lists campaigns in order x0, x1, ...; 'Qiskit string' is the same bitstring as Qiskit "
                   "prints it (qubit 0 on the right).")

# ----------------------------------------------------------------------------- tab: circuit
with tabs[5]:
    if not out:
        st.info("Run the optimization first (sidebar).")
    else:
        st.pyplot(out["circuit_fig"])
        st.markdown("""
| Element | Meaning |
|---|---|
| **Qubits** | One per campaign (qubit *i* = campaign *i*); state 1 means selected. Slack qubits appear only in QUBO mode. |
| **H (Hadamard)** | Puts all qubits in equal superposition: every combination is "tried" at once. |
| **Cost layer** | Applies e^(−iγH_C): adds a phase to each combination depending on its penalised cost (Z-rotations + CNOT ladders). |
| **Mixer layer** | RX(2β) on every qubit; mixes amplitudes so good solutions can interfere constructively. |
| **Parameterised rotations** | γ and β are tuned by the classical optimiser (COBYLA). |
| **Measurement** | Each shot yields one candidate selection; many shots give the probability distribution. |
""")
        st.caption("The cost layer is drawn as a single box for readability. The transpiled circuit that Aer runs is "
                   "much deeper (see 'QAOA details').")

# ----------------------------------------------------------------------------- tab: about
with tabs[6]:
    st.markdown("""
### What this project shows
A marketing budget-allocation problem is encoded as a binary optimisation problem, converted into a cost Hamiltonian,
and solved heuristically with **QAOA** on the **Qiskit Aer** simulator. Brute force gives the exact answer; greedy is a
fast classical heuristic.

### Honest limitations
* **QAOA is a heuristic.** It may not find the optimum and its result depends on p, penalty, optimiser, seed and shots.
* **Simulation, not hardware.** A classical simulator tracks 2^N amplitudes, so it cannot scale and offers no speed-up.
* **5 qubits do not prove quantum advantage.** 32 combinations are solved instantly by brute force; advantage is an
  asymptotic claim about large problems versus the best classical algorithms.
* **Higher-order terms.** The 5-qubit hinge penalty needs multi-qubit Z terms; real devices would pay for them in depth and noise.

### Tech stack
Python · Qiskit · Qiskit Aer · SciPy (COBYLA) · NumPy · Pandas · Matplotlib · Streamlit
""")
