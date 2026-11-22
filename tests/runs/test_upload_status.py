import importlib.util
import json
from pathlib import Path

from soamp.runs.spec import Experiment, pending_experiments

spec = importlib.util.spec_from_file_location("upload_status", Path(__file__).parents[2] / "scripts/exp/upload_status.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_record_has_cells_rc_and_log_tail(tmp_path, monkeypatch):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/vast_run_status.json").write_text(json.dumps({"c1": {"returncode": 0, "seconds": 5}}))
    (tmp_path / "reports/vast_remote.log").write_text("\n".join(f"l{i}" for i in range(300)))
    monkeypatch.setenv("SOAMP_EXPERIMENT", "e1")
    rec = mod.build_record(0, root=tmp_path)
    assert rec["experiment"] == "e1" and rec["rc"] == 0 and rec["cells"]["c1"]["seconds"] == 5
    assert len(rec["log_tail"]) == 100 and rec["log_tail"][-1] == "l299"


def test_record_when_runner_never_started(tmp_path):
    rec = mod.build_record(10, root=tmp_path)
    assert rec["rc"] == 10 and rec["cells"] == {} and rec["log_tail"] == []


def test_finished_wandb_runs_make_cells_not_pending(tmp_path):
    exps = [Experiment(Path("a.yaml"), "a", tmp_path / "ra.csv"), Experiment(Path("b.yaml"), "b", tmp_path / "rb.csv")]
    assert [e.exp_id for e in pending_experiments(exps, {"a"})] == ["b"]
    (tmp_path / "rb.csv").write_text("x")
    assert pending_experiments(exps, {"a"}) == []
    assert len(pending_experiments(exps)) == 1
