# Quantum Marketing Optimizer: Campaign Selection Using QAOA (with Web UI)

Select advertising campaigns under a fixed budget using **QAOA** (Qiskit + Qiskit Aer), and compare it honestly
with an **exact brute-force** search and a **greedy** heuristic. The project ships as:

* an interactive **web app** (Streamlit) for live demonstration, and
* a step-by-step **Jupyter / Colab notebook** that explains the maths and code.

```
app.py                         Streamlit UI
qmo/model.py                   Problem, brute force, greedy, result analysis (no Qiskit)
qmo/core.py                    Cost Hamiltonian, QAOA circuit, Aer simulation (Qiskit)
qmo/worker.py                  Persistent thread that owns Qiskit (see "Design notes")
qmo/plots.py                   Matplotlib figures
tests/                         20 automated tests (pytest)
notebooks/Quantum_Marketing_Optimizer.ipynb   Full explanatory notebook (saved with real outputs)
notebooks/Run_UI_in_Colab.ipynb               Launches the web app inside Google Colab
DEMO_GUIDE.md                  Suggested 5-minute demo script + likely questions
```

## 1. Run the web app on your computer (recommended for the demo)

Requires Python 3.9+ (tested on 3.12).

**Windows:** double-click `run_app.bat`  
**Mac / Linux:** `./run_app.sh`  
**Or manually:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Your browser opens at <http://localhost:8501>. Press **▶ Run optimization** in the sidebar.

## 2. Run the web app in Google Colab

Open `notebooks/Run_UI_in_Colab.ipynb` in Colab, run the cells, upload this ZIP when asked, and open the printed link.
*(This launcher relies on Colab's port-proxy feature and was not testable from the build environment, so try it before
the demo. The local method above is the safest option.)*

## 3. Run the notebook
Upload `notebooks/Quantum_Marketing_Optimizer.ipynb` to <https://colab.research.google.com> and use **Runtime → Run all**
(about a minute). The first cell installs `qiskit`, `qiskit-aer`, `pylatexenc`.

## What the UI offers
| Tab | Content |
|---|---|
| Problem | Editable campaign table (name, cost, return), budget, cost-vs-return chart |
| Formulation | Binary variables, objective, correct budget penalty, QUBO / slack discussion, verified penalty search |
| Results | Metric cards, comparison table (brute force / greedy / QAOA), over-budget rejection notice, random-sampling reality check |
| Charts | Selection heatmap, objective comparison, QAOA probability distribution, optional depth study (p = 1, 2, 3) |
| QAOA details | Optimiser convergence, angles, gate counts, every measured bitstring |
| Circuit | The real Qiskit circuit with explanation |
| About | Limitations and tech stack |

Sidebar controls: preset or custom data (2 to 8 campaigns), budget, budget encoding (5-qubit hinge or slack QUBO),
layers p, restarts, iterations, shots, seed.

## Maths in brief
* Variables `x_i ∈ {0,1}`; maximise `Σ r_i x_i` subject to `Σ c_i x_i ≤ B` (0/1 knapsack).
* The naive penalty `P(Σ c_i x_i − B)²` is **wrong** for an inequality: it also punishes under-spending. It is not used.
* **Mode A (default, 5 qubits):** hinge penalty `P·max(0, Σ c_i x_i − B)`. Correct for `≤`, no extra qubits, but not
  quadratic (its Z-expansion has up to 5-body terms, so it is a HUBO rather than a strict QUBO).
* **Mode B:** true QUBO with binary slack variables (`Σ c_i x_i + s = B`); needs ⌊log₂B⌋+1 extra qubits (4 slack qubits, so 9 qubits in total, for B = 10).
* **Penalty strength:** the smallest integer P that is *verified exhaustively* to keep the global minimum feasible and
  equal to the brute-force optimum (safe upper bound `1 + max r_i`).
* The Pauli-Z Hamiltonian is obtained by a Walsh-Hadamard transform and **checked** against the cost function with an
  independent label-based computation.
* Bit order: Qiskit is little-endian (qubit 0 = rightmost character); decoding is centralised and unit-tested.

## Honesty policy
* QAOA is a heuristic and may miss the optimum. The app says so whenever it does.
* The most frequent measured bitstring is shown unfiltered; if it is over budget it is flagged and **rejected**. The
  reported QAOA answer is the best *feasible* measured bitstring.
* A random-sampling baseline is shown, because with 32 outcomes and thousands of shots almost any sampler finds the optimum.
  The meaningful metrics are the probability on the optimum and the feasible share versus random.
* QAOA time is classical simulation time, not quantum hardware time. A 5-qubit simulation does **not** demonstrate
  quantum advantage.

## Tests
```bash
pip install pytest
python -m pytest -q tests        # 20 tests, about 20 seconds
```
They cover brute force against an independent reference, greedy invariants on random problems, little-endian decoding,
the Walsh-Hadamard transform, penalty minimality and validity, slack encodings, Hamiltonian correctness, QAOA output
consistency, input validation, qubit-limit protection, the thread-safety regression, and headless runs of the UI.

## Design notes
* **Persistent Qiskit worker thread.** Streamlit runs each script rerun in a new thread. During testing we found that
  Qiskit 2.x (Rust-backed) can crash the whole process with a segmentation fault if it is first imported in a thread that
  later exits and then used from another thread (reproducible with plain Python threads). The app therefore imports and
  uses Qiskit only inside one long-lived worker thread (`qmo/worker.py`), and the UI thread handles plain data only.
  `tests/test_worker.py` guards against regression.
* Limits: at most 8 campaigns and 12 qubits, to keep simulation fast.
* Saved notebook outputs come from one run (Qiskit 2.6.2, Aer 0.18.4). Seeds are fixed, but exact probabilities may differ
  slightly with other library versions.

## Dependencies
`qiskit>=1.2`, `qiskit-aer>=0.15`, `pylatexenc`, `numpy`, `pandas`, `scipy`, `matplotlib`, `streamlit>=1.50`
