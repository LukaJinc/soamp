import tarfile

import pytest

from soamp.runs.bundle import build_tarball, select_files
from soamp.runs.launch_plan import data_files


def test_select_keeps_code_and_config_drops_data_and_reports():
    files = ["src/soamp/a.py", "pipeline/train_cv.py", "config/train/x.yaml", "scripts/vast/remote_main.sh",
             "scripts/EDA/n.ipynb", "data/big.csv", "reports/cv_results_x.csv", "pyproject.toml", "presentations/a.pdf"]
    assert select_files(files, extra=["config/train/generated/e/c.yaml"]) == sorted(
        ["src/soamp/a.py", "pipeline/train_cv.py", "config/train/x.yaml", "scripts/vast/remote_main.sh", "pyproject.toml",
         "config/train/generated/e/c.yaml"])


def test_tarball_roundtrip_and_missing_file(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/a.yaml").write_text("x: 1")
    out = build_tarball(tmp_path, ["config/a.yaml"], tmp_path / "code.tar.gz")
    with tarfile.open(out) as t:
        assert t.getnames() == ["config/a.yaml"]
    with pytest.raises(FileNotFoundError):
        build_tarball(tmp_path, ["nope.txt"], tmp_path / "x.tar.gz")


def test_data_files_cover_required_and_produced_without_duplicates():
    f = data_files("dev")
    assert "data/dev/mic_classification_dataset.csv" in f and "data/dev/organism_vocab_vocab_embedding.json" in f
    assert len(f) == len(set(f))
