import pytest

from soamp.features.peptide import (
    DESCRIPTOR_NAMES,
    PeptideFeatureError,
    build_peptide_feature_rows,
    compute_peptide_descriptors,
    index_feature_rows_by_peptide_id,
    select_unique_peptides,
)

GLYCINE = "C(C(=O)O)N"
AMMONIUM = "[NH4+]"


def test_compute_descriptors_returns_none_for_invalid_smiles():
    assert compute_peptide_descriptors("not a smiles!!!") is None


def test_compute_descriptors_returns_none_for_empty_smiles():
    assert compute_peptide_descriptors("") is None


def test_compute_descriptors_returns_keys_in_declared_order():
    result = compute_peptide_descriptors(GLYCINE)
    assert list(result.keys()) == DESCRIPTOR_NAMES


def test_compute_descriptors_glycine_molwt_cross_checked_against_rdkit_directly():
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(GLYCINE)
    expected = Descriptors.MolWt(mol)
    result = compute_peptide_descriptors(GLYCINE)
    assert result["MolWt"] == pytest.approx(expected)


def test_compute_descriptors_formal_charge_on_ammonium_ion():
    result = compute_peptide_descriptors(AMMONIUM)
    assert result["FormalCharge"] == 1


def test_select_unique_peptides_dedups_by_peptide_id():
    rows = [
        {"peptide_id": "1", "smiles": "C", "organism": "Escherichia coli"},
        {"peptide_id": "1", "smiles": "C", "organism": "Staphylococcus aureus"},
        {"peptide_id": "2", "smiles": "CC", "organism": "Escherichia coli"},
    ]
    result = select_unique_peptides(rows)
    assert result == [
        {"peptide_id": "1", "smiles": "C"},
        {"peptide_id": "2", "smiles": "CC"},
    ]


def test_select_unique_peptides_raises_on_conflicting_smiles():
    rows = [
        {"peptide_id": "1", "smiles": "C", "organism": "Escherichia coli"},
        {"peptide_id": "1", "smiles": "CC", "organism": "Staphylococcus aureus"},
    ]
    with pytest.raises(PeptideFeatureError):
        select_unique_peptides(rows)


def test_build_peptide_feature_rows_end_to_end():
    unique_peptides = [{"peptide_id": "1", "smiles": GLYCINE}]
    result = build_peptide_feature_rows(unique_peptides)
    assert len(result) == 1
    assert result[0]["peptide_id"] == "1"
    assert set(result[0].keys()) == {"peptide_id", *DESCRIPTOR_NAMES}


def test_build_peptide_feature_rows_raises_on_unparseable_smiles():
    unique_peptides = [{"peptide_id": "1", "smiles": "not a smiles!!!"}]
    with pytest.raises(PeptideFeatureError):
        build_peptide_feature_rows(unique_peptides)


def test_index_feature_rows_by_peptide_id_reshapes_and_casts_to_float():
    feature_rows = [
        {"peptide_id": "1", "a": "1.5", "b": "2.5"},
        {"peptide_id": "2", "a": "3.0", "b": "4.0"},
    ]
    result = index_feature_rows_by_peptide_id(feature_rows, descriptor_names=["a", "b"])
    assert result == {"1": {"a": 1.5, "b": 2.5}, "2": {"a": 3.0, "b": 4.0}}


def test_index_feature_rows_by_peptide_id_raises_on_missing_descriptor():
    feature_rows = [{"peptide_id": "1", "a": "1.0"}]
    with pytest.raises(PeptideFeatureError):
        index_feature_rows_by_peptide_id(feature_rows, descriptor_names=["a", "b"])
