from pathlib import Path

import pytest
from pydantic import ValidationError

from soamp.runs.spec import (
    Experiment,
    RunSpec,
    pending_experiments,
    resolve_config_paths,
)


def _touch_configs(root: Path, names: list[str]) -> None:
    d = root / "config/train"
    d.mkdir(parents=True)
    for n in names:
        (d / n).write_text("exp_id: x\n")


def test_spec_requires_experiments_or_grid():
    with pytest.raises(ValidationError):
        RunSpec()


def test_spec_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        RunSpec(experiments=["a.yaml"], typo=1)


def test_grid_cross_product_with_exclude(tmp_path):
    _touch_configs(tmp_path, ["cv_a_x_long.yaml", "cv_a_y_long.yaml", "cv_b_x_long.yaml"])
    spec = RunSpec(grid={"peptide": ["a", "b"], "organism": ["x", "y"],
                         "variant": "_long", "exclude": [("b", "y")]})
    got = resolve_config_paths(spec, repo_root=tmp_path)
    assert [p.name for p in got] == ["cv_a_x_long.yaml", "cv_a_y_long.yaml", "cv_b_x_long.yaml"]


def test_missing_config_raises_listing_all(tmp_path):
    _touch_configs(tmp_path, [])
    spec = RunSpec(grid={"peptide": ["a"], "organism": ["x", "y"]})
    with pytest.raises(FileNotFoundError, match="cv_a_x.yaml.*cv_a_y.yaml"):
        resolve_config_paths(spec, repo_root=tmp_path)


def test_explicit_and_grid_are_deduplicated(tmp_path):
    _touch_configs(tmp_path, ["cv_a_x.yaml"])
    spec = RunSpec(experiments=["config/train/cv_a_x.yaml"],
                   grid={"peptide": ["a"], "organism": ["x"]})
    assert len(resolve_config_paths(spec, repo_root=tmp_path)) == 1


def test_pending_skips_finished(tmp_path):
    done = tmp_path / "cv_results_done.csv"
    done.write_text("x")
    exps = [Experiment(Path("a"), "done", done),
            Experiment(Path("b"), "todo", tmp_path / "cv_results_todo.csv")]
    assert [e.exp_id for e in pending_experiments(exps)] == ["todo"]


def test_prepare_steps_pending_and_required_artifacts(tmp_path):
    from soamp.runs.spec import pending_prepare_steps, required_artifacts_missing

    spec = RunSpec(
        experiments=["a.yaml"],
        required_artifacts=["data/x.csv", "data/feat.csv"],
        prepare=[{"cmd": ["echo"], "produces": ["data/feat.csv"]}],
    )
    assert len(pending_prepare_steps(spec, repo_root=tmp_path)) == 1
    # a prepared artifact is not "missing"; an unprepared absent one is
    assert [str(p) for p in required_artifacts_missing(spec, repo_root=tmp_path)] == ["data/x.csv"]
    (tmp_path / "data").mkdir()
    (tmp_path / "data/feat.csv").write_text("x")
    assert pending_prepare_steps(spec, repo_root=tmp_path) == []


def test_offer_query_uses_spec_filters():
    from soamp.runs.spec import InstanceSpec, offer_query

    q = offer_query(InstanceSpec(gpu_name="A100_SXM4", disk_gb=60, min_reliability=0.995))
    assert "gpu_name=A100_SXM4" in q and "reliability>0.995" in q and "verified=true" in q
    assert "disk_space>=60" in q
