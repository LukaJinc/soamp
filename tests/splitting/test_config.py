import pytest
from pydantic import ValidationError

from soamp.splitting.config import SplittingConfig
from soamp.utils.config import load_config


def test_base_yaml_loads():
    cfg = load_config("config/splitting/base.yaml", SplittingConfig)
    assert cfg.fingerprint_graph.fingerprint_threshold == 0.8
    assert cfg.fingerprint_graph.radius == 2
    assert cfg.fingerprint_graph.n_bits == 2048
    assert cfg.qmap_graph.identity_threshold == 0.60
    assert cfg.qmap_graph.matrix == "blosum45"
    assert cfg.qmap_graph.gap_open == 5
    assert cfg.qmap_graph.gap_extension == 1
    assert cfg.qmap_graph.use_cache is True
    assert cfg.qmap_graph.num_threads is None
    assert cfg.leiden.n_iterations == 2
    assert cfg.leiden.seed == 42
    assert cfg.bucketing.test_size == 0.2
    assert cfg.bucketing.n_folds == 5
    assert cfg.bucketing.lambda_noncanonical == 8.0
    assert cfg.bucketing.n_iterations == 40_000
    assert cfg.bucketing.seed == 42
    assert cfg.output.filename == "peptide_split.csv"
    assert cfg.output.sidecar_filename == "peptide_split.json"
    assert cfg.paths.data_dir.name == "data"
    assert cfg.paths.data_dir.is_absolute()


def test_extra_top_level_key_rejected():
    with pytest.raises(ValidationError):
        SplittingConfig.model_validate({"bogus_top_level_key": 1})


def test_extra_nested_key_rejected():
    with pytest.raises(ValidationError):
        SplittingConfig.model_validate({"fingerprint_graph": {"bogus_key": "x"}})
    with pytest.raises(ValidationError):
        SplittingConfig.model_validate({"qmap_graph": {"bogus_key": "x"}})
    with pytest.raises(ValidationError):
        SplittingConfig.model_validate({"leiden": {"bogus_key": "x"}})
    with pytest.raises(ValidationError):
        SplittingConfig.model_validate({"bucketing": {"bogus_key": "x"}})


def test_base_override_composition(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "paths:\n  data_dir: data\n"
        "fingerprint_graph:\n  fingerprint_threshold: 0.8\n"
    )
    override = tmp_path / "experiment.yaml"
    override.write_text(
        "_base_: base.yaml\n"
        "fingerprint_graph:\n  fingerprint_threshold: 0.9\n"
    )
    cfg = load_config(override, SplittingConfig)
    assert cfg.fingerprint_graph.fingerprint_threshold == 0.9
    assert cfg.paths.data_dir.name == "data"
