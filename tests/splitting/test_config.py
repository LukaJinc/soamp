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
    assert cfg.bucketing.lambda_noncanonical == 30.0
    assert cfg.bucketing.lambda_active == 30.0
    assert cfg.bucketing.n_iterations == 100_000
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


# ---- dev-split config -----------------------------------------------------

def test_dev_configs_load_and_differ_only_where_intended():
    from soamp.splitting.dev_config import DevSplitConfig
    from soamp.utils.config import load_config

    v1 = load_config("config/splitting/dev.yaml", DevSplitConfig)
    v2 = load_config("config/splitting/dev_v2.yaml", DevSplitConfig)
    assert (v1.identity_threshold, v1.d_residues) == (0.6, "separate")
    assert (v2.identity_threshold, v2.d_residues) == (0.8, "map_to_L")
    assert v1.hard_link_identity == v2.hard_link_identity == 0.9
    assert v1.organisms == v2.organisms and len(v1.organisms) == 7
    assert v1.split_output != v2.split_output           # v2 never overwrites v1
    assert "dbaasp_dev_split.parquet" not in {v1.split_output.name, v2.split_output.name}   # the first (3-strain) split is never a target
    assert "D-TIC" in v2.name_exceptions


def test_dev_config_rejects_unknown_key_and_bad_values():
    import pytest
    from pydantic import ValidationError

    from soamp.splitting.dev_config import DevSplitConfig

    base = dict(organisms=["x"], identity_threshold=0.8, split_output="a.parquet")
    DevSplitConfig(**base)
    with pytest.raises(ValidationError):
        DevSplitConfig(**base, d_residue="map_to_L")        # typo'd key
    with pytest.raises(ValidationError):
        DevSplitConfig(**{**base, "d_residues": "mirror"})
    with pytest.raises(ValidationError):
        DevSplitConfig(**{**base, "identity_threshold": 80})
