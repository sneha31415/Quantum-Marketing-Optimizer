# Demo Guide (about 5 minutes)

## Before you start
1. Run the app (`run_app.bat` / `./run_app.sh`) and click **▶ Run optimization** once so everything is warm.
2. Keep the notebook open in another tab in case you are asked to show code.

## Suggested flow
1. **Problem tab (30 s).** "A marketing team has 5 campaigns, each with a cost and expected return, and a budget of 10.
   Pick the best combination. That is a 0/1 knapsack problem; 5 campaigns give 2⁵ = 32 combinations."
2. **Formulation tab (60 s).** Show the two penalty formulas. Key point: "The textbook penalty (Σcx − B)² is wrong for a *≤* budget,
   because it also punishes spending less. We either use a one-sided hinge (5 qubits) or slack variables (9 qubits, a true QUBO)."
   Point to the **penalty table**: P = 3 fails (an over-budget selection wins), P = 4 is the smallest valid one.
3. **Results tab (90 s).** Brute force is optimal (return 15). **Greedy gets 14**, so a fast classical heuristic misses it.
   QAOA's best feasible sample reaches 15. Then the important honesty point: the **most frequent** bitstring was
   `10100` which is **over budget** (cost 11); the app rejects it. Show the **reality check**: random sampling also finds 15
   with 8192 shots, so the meaningful numbers are P(optimum) ≈ 9.7% vs 3.1% random (about 3× lift).
4. **Charts tab (45 s).** Probability distribution: green = feasible, red = over budget, outlined = true optimum.
   Tick **depth study** in the sidebar and re-run to show p = 1, 2, 3.
5. **Circuit tab (45 s).** H on every qubit → cost layer (phase by cost) → RX mixer → measurement; γ and β are tuned by COBYLA.
6. **Conclusion (30 s).** "QAOA is a heuristic; this is a classical simulation, so it demonstrates the method, not quantum advantage."

## Likely questions from the examiner
**Why only 5 qubits?** One qubit per campaign keeps the simulation instant and the circuit drawable. The app supports up to 8 campaigns.

**Is this faster than classical?** No. The simulator tracks 2ⁿ amplitudes on a classical CPU. Quantum advantage is an
asymptotic claim for large problems; 32 combinations are trivial for brute force.

**Why did QAOA not always find the optimum / why is P(optimum) only ~10%?** It is a heuristic, with shallow depth (p = 2),
a penalised landscape and shot noise. In an exact (noise-free, infinite-shot) statevector check during development, p = 2 reached only about 8 to 10% on this problem, so the measured ~10% is in line with what the method can do here; more layers help only slightly.

**Why not just use (Σcx − B)²?** It is an equality constraint; it rewards spending exactly B.

**How did you choose the penalty?** The app tries P = 1, 2, 3, … and picks the smallest value verified exhaustively
to keep the global minimum feasible and optimal. Too large a P drowns the return signal.

**What is QUBO vs HUBO?** QUBO has only quadratic terms (needs slack qubits for ≤). Our 5-qubit hinge has higher-order terms (HUBO).

**What if QAOA returns an infeasible answer?** It is detected and rejected; only feasible bitstrings are reported as solutions.

**What does COBYLA do?** A gradient-free classical optimiser that adjusts γ and β to minimise the expected cost measured on the simulator.

**Which Qiskit bit order?** Little-endian: qubit 0 is the rightmost character. All decoding goes through one tested function.
