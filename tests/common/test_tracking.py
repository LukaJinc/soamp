import json
import os
from unittest.mock import MagicMock

import pytest
import wandb

from soamp.common.tracking import (
    LocalFileTracker,
    TrackingBackendError,
    TrackingConfig,
    WandbTracker,
    build_tracker,
)


def test_log_scalar_appends_jsonl_line(tmp_path):
    tracker = LocalFileTracker(tmp_path, "exp_a")
    tracker.log_scalar("train_loss", 0.5, step=1)
    tracker.log_scalar("train_loss", 0.4, step=2)
    lines = (tmp_path / "exp_a" / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["name"] == "train_loss"
    assert record["value"] == 0.5
    assert record["step"] == 1


def test_log_metrics_writes_one_line_per_metric_same_step(tmp_path):
    tracker = LocalFileTracker(tmp_path, "exp_a")
    tracker.log_metrics({"accuracy": 0.9, "f1": 0.8}, step=1)
    lines = (tmp_path / "exp_a" / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert {r["name"] for r in records} == {"accuracy", "f1"}
    assert all(r["step"] == 1 for r in records)


def test_log_config_writes_run_metadata_json(tmp_path):
    tracker = LocalFileTracker(tmp_path, "exp_a")
    tracker.log_config({"seed": 42, "lr": 0.001})
    metadata = json.loads((tmp_path / "exp_a" / "run_metadata.json").read_text())
    assert metadata == {"seed": 42, "lr": 0.001}


def test_run_dir_created_per_exp_id(tmp_path):
    LocalFileTracker(tmp_path, "exp_a")
    LocalFileTracker(tmp_path, "exp_b")
    assert (tmp_path / "exp_a").is_dir()
    assert (tmp_path / "exp_b").is_dir()


def test_log_artifact_writes_manifest_with_file_hash(tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_text("a,b\n1,2\n")

    tracker = LocalFileTracker(tmp_path / "runs", "exp_a")
    manifest = tracker.log_artifact(
        name="my_dataset", artifact_type="dataset", paths=[data_file], metadata={"n_rows": 1}
    )

    manifest_path = tmp_path / "runs" / "exp_a" / "artifacts" / "my_dataset.json"
    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk["name"] == "my_dataset"
    assert on_disk["type"] == "dataset"
    assert on_disk["metadata"] == {"n_rows": 1}
    assert len(on_disk["files"]) == 1
    assert on_disk["files"][0]["path"] == str(data_file)
    assert len(on_disk["files"][0]["sha256"]) == 64
    assert manifest == on_disk


def test_log_artifact_records_depends_on_by_name(tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_text("x")
    ckpt_file = tmp_path / "model.pth"
    ckpt_file.write_text("y")

    tracker = LocalFileTracker(tmp_path / "runs", "exp_a")
    dataset_artifact = tracker.log_artifact("my_dataset", "dataset", [data_file])
    model_artifact = tracker.log_artifact(
        "my_model", "model", [ckpt_file], depends_on=[dataset_artifact]
    )

    assert model_artifact["depends_on"] == ["my_dataset"]


def test_log_artifact_records_depends_on_by_bare_string(tmp_path):
    ckpt_file = tmp_path / "model.pth"
    ckpt_file.write_text("y")

    tracker = LocalFileTracker(tmp_path / "runs", "exp_a")
    model_artifact = tracker.log_artifact(
        "my_model", "model", [ckpt_file], depends_on=["upstream_from_another_process"]
    )
    assert model_artifact["depends_on"] == ["upstream_from_another_process"]


def test_log_artifact_hashes_every_file_in_a_directory(tmp_path):
    d = tmp_path / "features"
    d.mkdir()
    (d / "a.csv").write_text("a")
    (d / "b.csv").write_text("b")

    tracker = LocalFileTracker(tmp_path / "runs", "exp_a")
    manifest = tracker.log_artifact("features", "dataset", [d])
    assert len(manifest["files"]) == 2


def test_build_tracker_local_backend_returns_local_file_tracker(tmp_path):
    cfg = TrackingConfig(backend="local", exp_id="x")
    tracker = build_tracker(cfg, tmp_path)
    assert isinstance(tracker, LocalFileTracker)
    assert (tmp_path / "x").is_dir()


def test_build_tracker_wandb_backend_returns_wandb_tracker(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    cfg = TrackingConfig(backend="wandb", exp_id="x", wandb_project="p")
    tracker = build_tracker(cfg, tmp_path)
    assert isinstance(tracker, WandbTracker)
    tracker.close()


# WandbTracker tests use mode="disabled" -- wandb.init() becomes a local
# no-op with no network calls and no valid API key required.


def test_wandb_tracker_log_config_does_not_raise():
    tracker = WandbTracker(project="soamp-test", exp_id="test_exp", mode="disabled")
    tracker.log_config({"seed": 42})
    tracker.close()


def test_wandb_tracker_log_scalar_does_not_raise():
    tracker = WandbTracker(project="soamp-test", exp_id="test_exp", mode="disabled")
    tracker.log_scalar("train_loss", 0.5, step=1)
    tracker.close()


def test_wandb_tracker_log_metrics_does_not_raise():
    tracker = WandbTracker(project="soamp-test", exp_id="test_exp", mode="disabled")
    tracker.log_metrics({"accuracy": 0.9, "f1": 0.8}, step=1)
    tracker.close()


def test_wandb_tracker_implements_experiment_tracker_interface():
    tracker = WandbTracker(project="soamp-test", exp_id="test_exp", mode="disabled")
    for method in ("log_config", "log_scalar", "log_metrics", "log_artifact", "close"):
        assert callable(getattr(tracker, method))
    tracker.close()


def test_wandb_tracker_log_artifact_does_not_raise(tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_text("a,b\n1,2\n")

    tracker = WandbTracker(project="soamp-test", exp_id="test_exp", mode="disabled")
    artifact = tracker.log_artifact("my_dataset", "dataset", [data_file], metadata={"n_rows": 1})
    assert artifact.type == "dataset"
    tracker.close()


def test_wandb_tracker_log_artifact_with_depends_on_does_not_raise(tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_text("a")
    ckpt_file = tmp_path / "model.pth"
    ckpt_file.write_text("b")

    tracker = WandbTracker(project="soamp-test", exp_id="test_exp", mode="disabled")
    dataset_artifact = tracker.log_artifact("my_dataset", "dataset", [data_file])
    model_artifact = tracker.log_artifact(
        "my_model", "model", [ckpt_file], depends_on=[dataset_artifact]
    )
    assert model_artifact.type == "model"
    tracker.close()


def test_wandb_tracker_log_artifact_bare_string_depends_on_gets_latest_suffix(tmp_path):
    ckpt_file = tmp_path / "model.pth"
    ckpt_file.write_text("b")

    tracker = WandbTracker(project="soamp-test", exp_id="test_exp", mode="disabled")
    tracker._run.use_artifact = MagicMock()
    tracker.log_artifact("my_model", "model", [ckpt_file], depends_on=["upstream_dataset"])
    tracker._run.use_artifact.assert_called_once_with("upstream_dataset:latest")
    tracker.close()


def test_wandb_tracker_log_artifact_versioned_string_depends_on_passed_through(tmp_path):
    ckpt_file = tmp_path / "model.pth"
    ckpt_file.write_text("b")

    tracker = WandbTracker(project="soamp-test", exp_id="test_exp", mode="disabled")
    tracker._run.use_artifact = MagicMock()
    tracker.log_artifact("my_model", "model", [ckpt_file], depends_on=["upstream_dataset:v3"])
    tracker._run.use_artifact.assert_called_once_with("upstream_dataset:v3")
    tracker.close()


def test_wandb_tracker_log_artifact_in_process_handle_depends_on_passed_as_object(tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_text("a")
    ckpt_file = tmp_path / "model.pth"
    ckpt_file.write_text("b")

    tracker = WandbTracker(project="soamp-test", exp_id="test_exp", mode="disabled")
    dataset_artifact = tracker.log_artifact("my_dataset", "dataset", [data_file])
    tracker._run.use_artifact = MagicMock()
    tracker.log_artifact("my_model", "model", [ckpt_file], depends_on=[dataset_artifact])
    tracker._run.use_artifact.assert_called_once_with(dataset_artifact)
    tracker.close()
