import sys
import types

from soamp.runs.cell_reports import cell_report_files, upload_cell_reports

EXP = "e1__mlp__rdkit_vocab__abc123"


def test_cell_report_files_finds_only_this_cells_files(tmp_path):
    reports = tmp_path / "reports"
    (reports / "dev_v2").mkdir(parents=True)
    mine = [reports / f"train_cv_{EXP}_log.txt", reports / "dev_v2" / f"cv_results_{EXP}.csv"]
    for p in mine + [reports / "cv_results_e1__mlp__rdkit_vocab__ffffff.csv", reports / "vast_run_status.json"]:
        p.write_text("x")
    assert cell_report_files(reports, EXP) == sorted(mine)
    assert cell_report_files(tmp_path / "absent", EXP) == []
    assert cell_report_files(reports, "") == []


def _fake_wandb(monkeypatch):
    calls: dict = {"files": [], "finished": 0}

    class Run:
        def log_artifact(self, art):
            calls["logged"] = art
            return types.SimpleNamespace(wait=lambda: None)

        def finish(self):
            calls["finished"] += 1

    class Artifact:
        def __init__(self, name, type, metadata):
            self.name, self.type, self.metadata = name, type, metadata
            self.qualified_name = f"ent/soamp-ops/{name}:v0"

        def add_file(self, path, name):
            calls["files"].append(name)

    def init(**kw):
        calls["init"] = kw
        return Run()

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init, Artifact=Artifact))
    return calls


def test_upload_logs_one_ops_artifact_per_cell_with_repo_relative_paths(tmp_path, monkeypatch):
    calls = _fake_wandb(monkeypatch)
    (tmp_path / "reports").mkdir()
    f = tmp_path / "reports" / f"train_cv_{EXP}_log.txt"
    f.write_text("log")
    ref = upload_cell_reports(EXP, [f], repo_root=tmp_path, group="e1", entity="ent", returncode=1)
    assert ref == f"ent/soamp-ops/cell_reports-{EXP}:v0"
    assert calls["init"]["project"] == "soamp-ops" and calls["init"]["job_type"] == "cell_reports"
    assert calls["init"]["group"] == "e1"
    assert calls["files"] == [f"reports/train_cv_{EXP}_log.txt"]
    assert calls["logged"].metadata["returncode"] == 1
    assert calls["finished"] == 1


def test_upload_with_no_files_does_not_touch_wandb(tmp_path, monkeypatch):
    calls = _fake_wandb(monkeypatch)
    assert upload_cell_reports(EXP, [], repo_root=tmp_path, group="e1", entity=None, returncode=0) is None
    assert "init" not in calls
