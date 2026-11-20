import sys, pathlib
from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")


def fresh():
    sys.path.insert(0, str(ROOT))
    return AppTest.from_file(APP, default_timeout=180)


def click_run(at):
    [b for b in at.button if "Run optimization" in b.label][0].click()
    return at.run()


def test_app_loads_without_run():
    at = fresh().run()
    assert not at.exception
    assert any("Run optimization" in i.value for i in at.info)


def test_full_run_default_mode():
    at = click_run(fresh().run())
    assert not at.exception, at.exception
    assert not at.error
    labels = [m.label for m in at.metric]
    assert "Optimal return (brute force)" in labels
    vals = {m.label: m.value for m in at.metric}
    assert vals["Optimal return (brute force)"] == "15"
    assert vals["Greedy return"] == "14"
    assert vals["Qubits"] == "5"
    assert len(at.dataframe) >= 3


def test_slack_mode_and_depth_study():
    at = fresh().run()
    at.radio[0].set_value("Slack-variable QUBO (extra qubits)")
    at.slider[0].set_value(1)          # p = 1 to keep the test quick
    at.slider[1].set_value(2)          # restarts
    at.checkbox[0].set_value(True)     # depth study
    at = click_run(at)
    assert not at.exception, at.exception
    assert {m.label: m.value for m in at.metric}["Qubits"] == "9"


def test_invalid_input_shows_error_not_crash():
    at = fresh().run()
    at.number_input[0].set_value(1)    # budget 1: no campaign fits; empty selection is the only feasible one
    at = click_run(at)
    assert not at.exception, at.exception
