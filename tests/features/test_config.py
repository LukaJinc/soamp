import pytest
from pydantic import ValidationError

from soamp.features.config import FeaturesConfig
from soamp.features.peptide import DESCRIPTOR_NAMES
from soamp.utils.config import load_config


def test_base_yaml_loads():
    cfg = load_config("config/features/base.yaml", FeaturesConfig)
    assert cfg.peptide_featurization.method == "rdkit_descriptors"
    assert cfg.peptide_featurization.rdkit_descriptors.descriptor_names == DESCRIPTOR_NAMES
    assert cfg.organism_featurization.method == "vocab_embedding"
    assert cfg.organism_featurization.vocab_embedding.unknown_index == 0
    assert cfg.organism_featurization.kmer_composition.k_values == [1, 2, 3, 4]
    assert cfg.paths.data_dir.is_absolute()


def test_extra_key_rejected():
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate({"bogus_top_level_key": 1})


def test_extra_nested_key_rejected():
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate({"paths": {"bogus_key": "x"}})


def test_unknown_descriptor_name_rejected():
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate(
            {"peptide_featurization": {"rdkit_descriptors": {"descriptor_names": ["NotARealDescriptor"]}}}
        )


def test_unknown_peptide_method_rejected():
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate({"peptide_featurization": {"method": "not_a_real_method"}})


def test_unknown_organism_method_rejected():
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate({"organism_featurization": {"method": "not_a_real_method"}})


def test_peptide_peptideclm_yaml_loads_with_peptideclm_method():
    cfg = load_config("config/features/peptide_peptideclm.yaml", FeaturesConfig)
    assert cfg.peptide_featurization.method == "peptideclm_embedding"
    assert cfg.peptide_featurization.peptideclm_embedding.checkpoint == "aaronfeller/PeptideCLM-23M-all"


def test_kmer_organism_yaml_loads_with_kmer_composition_method():
    cfg = load_config("config/features/organism_kmer.yaml", FeaturesConfig)
    assert cfg.organism_featurization.method == "kmer_composition"
    assert cfg.organism_featurization.kmer_composition.k_values == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "overlay, field, expected",
    [
        ("peptide_rdkit", "peptide_features_filename", "peptide_features_rdkit.csv"),
        ("peptide_rdkit", "peptide_feature_scaler_filename", "peptide_feature_scaler_rdkit.json"),
        ("peptide_peptideclm", "peptide_features_filename", "peptide_features_peptideclm.csv"),
        ("peptide_peptideclm", "peptide_feature_scaler_filename",
         "peptide_feature_scaler_peptideclm.json"),
        ("organism_vocab", "organism_vocab_filename", "organism_vocab_vocab_embedding.json"),
        ("organism_kmer", "organism_vocab_filename", "organism_vocab_kmer_composition.json"),
    ],
)
def test_grid_overlays_write_method_distinct_artifact_filenames(overlay, field, expected):
    """The four grid cells' artifacts have to coexist in data/. base.yaml's
    unsuffixed filenames would have each features run silently overwrite the
    previous one's output."""
    cfg = load_config(f"config/features/{overlay}.yaml", FeaturesConfig)
    assert getattr(cfg.output_files, field) == expected


def test_base_override_composition(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "peptide_featurization:\n"
        "  rdkit_descriptors:\n"
        "    descriptor_names: [MolWt, TPSA]\n"
    )
    override = tmp_path / "experiment.yaml"
    override.write_text(
        "_base_: base.yaml\n"
        "peptide_featurization:\n"
        "  rdkit_descriptors:\n"
        "    descriptor_names: [MolWt]\n"
    )
    cfg = load_config(override, FeaturesConfig)
    assert cfg.peptide_featurization.rdkit_descriptors.descriptor_names == ["MolWt"]
