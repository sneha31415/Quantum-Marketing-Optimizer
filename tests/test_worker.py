"""Regression test for the Qiskit-in-short-lived-threads segfault (see qmo/worker.py)."""
import sys, pathlib, threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qmo import model  # noqa: E402
from qmo.worker import QuantumWorker  # noqa: E402


def test_worker_serves_many_short_lived_threads_without_crashing():
    w = QuantumWorker()
    p = model.Problem(["a", "b", "c", "d", "e"], [6, 5, 5, 3, 2], [10, 8, 7, 4, 2], 10)
    bf = model.brute_force(p)
    results, errors = [], []

    def job():
        try:
            cm = w.call("build_cost_model", p, "hinge5", opt_return=bf.ret)
            res = w.call("run_qaoa", p, cm, p=1, restarts=1, maxiter=15, shots_final=512, seed=1)
            fig = w.call("draw_circuit", cm, res)
            results.append((res.shots, fig is not None))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    for _ in range(3):                      # each call comes from a NEW thread that then exits (like Streamlit reruns)
        t = threading.Thread(target=job)
        t.start(); t.join()
    assert not errors and len(results) == 3 and all(r == (512, True) for r in results)


def test_worker_propagates_errors():
    w = QuantumWorker()
    p = model.Problem(["a", "b", "c", "d", "e", "f", "g", "h"], [4] * 8, [5] * 8, 500)
    try:
        w.call("build_cost_model", p, "slack_qubo")
        assert False, "should have raised"
    except ValueError as e:
        assert "qubits" in str(e)
