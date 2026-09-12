import csv
import json

import pytest

from soamp.data.factory import DatasetFactoryError, build_dataset
from soamp.features.peptide import DESCRIPTOR_NAMES
from soamp.features.scaling import fit_scaler

GLYCINE = "C(C(=O)O)N"
AMMONIUM = "[NH4+]"
ACETIC_ACID = "CC(=O)O"


def _fit_rows():
    return [
        {"peptide_id": "1", "smiles": GLYCINE, "organism": "Escherichia coli", "label": "active"},
        {"peptide_id": "2", "smiles": AMMONIUM, "organism": "Staphylococcus aureus", "label": "inactive"},
    ]


def _val_rows():
    return [
        {"peptide_id": "3", "smiles": ACETIC_ACID, "organism": "Pseudomonas aeruginosa", "label": "active"},
    ]


def test_from_rows_builds_one_dataset_per_group_key():
    bundle = build_dataset(row_groups={"fit": _fit_rows(), "val": _val_rows()})
    assert bundle.datasets.keys() == {"fit", "val"}
    assert len(bundle.datasets["fit"]) == 2
    assert len(bundle.datasets["val"]) == 1


def test_from_rows_supports_arbitrary_group_names():
    bundle = build_dataset(row_groups={"fit": _fit_rows(), "eval_other_pop": _val_rows()})
    assert bundle.datasets.keys() == {"fit", "eval_other_pop"}


def test_from_rows_raises_when_row_groups_empty():
    with pytest.raises(DatasetFactoryError):
        build_dataset(row_groups={})


def test_from_rows_raises_when_fit_group_missing():
    with pytest.raises(DatasetFactoryError):
        build_dataset(row_groups={"other": _fit_rows()}, fit_group="fit")


def test_from_rows_scaler_fit_only_on_fit_group_peptides():
    bundle = build_dataset(row_groups={"fit": _fit_rows(), "val": _val_rows()})

    from soamp.features.peptide import build_peptide_feature_rows, select_unique_peptides

    expected_scaler = fit_scaler(
        build_peptide_feature_rows(select_unique_peptides(_fit_rows())), DESCRIPTOR_NAMES
    )
    assert bundle.featurization.scaler["mean"] == pytest.approx(expected_scaler["mean"])
    assert bundle.featurization.scaler["scale"] == pytest.approx(expected_scaler["scale"])


def test_from_rows_vocab_only_includes_fit_group_organisms():
    bundle = build_dataset(row_groups={"fit": _fit_rows(), "val": _val_rows()})
    assert set(bundle.featurization.organism_vocab.keys()) == {
        "Escherichia coli", "Staphylococcus aureus",
    }
    # "Pseudomonas aeruginosa" only appears in val -> falls back to unknown_index
    _, organism_idx, _ = bundle.datasets["val"][0]
    assert organism_idx.item() == bundle.featurization.unknown_index


def test_from_rows_peptide_feature_dim_and_vocab_size_metadata_correct():
    bundle = build_dataset(row_groups={"fit": _fit_rows(), "val": _val_rows()})
    assert bundle.featurization.peptide_feature_dim == len(DESCRIPTOR_NAMES)
    assert bundle.featurization.organism_vocab_size == len(bundle.featurization.organism_vocab) + 1


def test_from_rows_respects_custom_descriptor_names():
    bundle = build_dataset(
        row_groups={"fit": _fit_rows(), "val": _val_rows()},
        peptide_method_kwargs={"descriptor_names": ["MolWt", "TPSA"]},
    )
    assert bundle.featurization.descriptor_names == ["MolWt", "TPSA"]
    assert bundle.featurization.peptide_feature_dim == 2
    features, _, _ = bundle.datasets["fit"][0]
    assert features.shape == (2,)


def test_from_rows_propagates_unparseable_smiles():
    from soamp.features.peptide import PeptideFeatureError

    bad_rows = [{"peptide_id": "1", "smiles": "not a smiles!!!", "organism": "Escherichia coli", "label": "active"}]
    with pytest.raises(PeptideFeatureError):
        build_dataset(row_groups={"fit": bad_rows})


def test_from_rows_defaults_organism_output_kind_to_index():
    bundle = build_dataset(row_groups={"fit": _fit_rows(), "val": _val_rows()})
    assert bundle.featurization.organism_output_kind == "index"
    assert bundle.featurization.organism_feature_dim is None
    assert bundle.featurization.peptide_method == "rdkit_descriptors"
    assert bundle.featurization.organism_method == "vocab_embedding"


def _write_artifact_fixture(tmp_path):
    classification_rows = [
        {"peptide_id": "1", "organism": "Escherichia coli", "label": "active", "split": "train"},
        {"peptide_id": "2", "organism": "Escherichia coli", "label": "inactive", "split": "train"},
        {"peptide_id": "3", "organism": "Escherichia coli", "label": "active", "split": "test"},
    ]
    with open(tmp_path / "mic_classification_dataset.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["peptide_id", "organism", "label", "split"])
        writer.writeheader()
        writer.writerows(classification_rows)

    feature_rows = [
        {"peptide_id": "1", "a": "1.0", "b": "10.0"},
        {"peptide_id": "2", "a": "2.0", "b": "20.0"},
        {"peptide_id": "3", "a": "3.0", "b": "30.0"},
    ]
    with open(tmp_path / "peptide_features.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["peptide_id", "a", "b"])
        writer.writeheader()
        writer.writerows(feature_rows)

    with open(tmp_path / "organism_vocab.json", "w") as f:
        json.dump({
            "method": "vocab_embedding", "unknown_index": 0,
            "vocab_size": 2, "vocab": {"Escherichia coli": 1},
        }, f)

    with open(tmp_path / "peptide_feature_scaler.json", "w") as f:
        json.dump({
            "method": "rdkit_descriptors",
            "descriptor_names": ["a", "b"], "mean": [0.0, 0.0], "scale": [1.0, 1.0],
        }, f)

    with open(tmp_path / "val_split.json", "w") as f:
        json.dump({"val_peptide_ids": ["2"]}, f)


def test_from_artifacts_default_reproduces_fit_val_test_keys(tmp_path):
    _write_artifact_fixture(tmp_path)
    bundle = build_dataset(data_dir=tmp_path)
    assert bundle.datasets.keys() == {"fit", "val", "test"}
    assert len(bundle.datasets["fit"]) == 1
    assert len(bundle.datasets["val"]) == 1
    assert len(bundle.datasets["test"]) == 1
    assert bundle.featurization.descriptor_names == ["a", "b"]
    assert bundle.featurization.peptide_feature_dim == 2
    assert bundle.featurization.organism_vocab_size == 2


def test_from_artifacts_defaults_organism_output_kind_to_index(tmp_path):
    _write_artifact_fixture(tmp_path)
    bundle = build_dataset(data_dir=tmp_path)
    assert bundle.featurization.organism_output_kind == "index"
    assert bundle.featurization.organism_feature_dim is None
    assert bundle.featurization.peptide_method == "rdkit_descriptors"
    assert bundle.featurization.organism_method == "vocab_embedding"


def test_from_artifacts_reconstructs_kmer_composition_organism_method(tmp_path):
    _write_artifact_fixture(tmp_path)
    with open(tmp_path / "organism_vocab.json", "w") as f:
        json.dump({
            "method": "kmer_composition", "k_values": [1, 2], "feature_dim": 20,
            "species_features": {"Escherichia coli": [0.1] * 20},
            "genus_features": {},
        }, f)

    bundle = build_dataset(data_dir=tmp_path)
    assert bundle.featurization.organism_output_kind == "vector"
    assert bundle.featurization.organism_feature_dim == 20
    assert bundle.featurization.organism_vocab_size is None
    assert bundle.featurization.organism_method == "kmer_composition"

    features, organism_vec, _ = bundle.datasets["fit"][0]
    assert organism_vec.shape == (20,)


def test_artifact_mode_rejects_peptide_method_override(tmp_path):
    _write_artifact_fixture(tmp_path)
    with pytest.raises(DatasetFactoryError):
        build_dataset(data_dir=tmp_path, peptide_method="peptideclm_embedding")


def test_artifact_mode_rejects_peptide_method_kwargs(tmp_path):
    _write_artifact_fixture(tmp_path)
    with pytest.raises(DatasetFactoryError):
        build_dataset(data_dir=tmp_path, peptide_method_kwargs={"descriptor_names": ["a"]})


def test_artifact_mode_rejects_organism_method_kwargs(tmp_path):
    _write_artifact_fixture(tmp_path)
    with pytest.raises(DatasetFactoryError):
        build_dataset(data_dir=tmp_path, organism_method_kwargs={"unknown_index": 1})
