import pytest

from soamp.data.assembly import (
    FIELDNAMES,
    DatasetAssemblyError,
    attach_split,
    build_classification_dataset,
    filter_labels,
    join_regression_and_labels,
)


def _label_row(peptide_id, organism, label):
    return {"peptide_id": peptide_id, "organism": organism, "label": label}


def _regression_row(peptide_id, organism, sequence="AAA", smiles="C", mic_value_uM="10.0",
                     mic_type="exact", taxon_id="562"):
    return {
        "peptide_id": peptide_id,
        "organism": organism,
        "sequence": sequence,
        "smiles": smiles,
        "ncbi_taxon_id_if_available": taxon_id,
        "mic_value_uM": mic_value_uM,
        "mic_type": mic_type,
    }


def test_filter_labels_keeps_only_included_labels():
    rows = [
        _label_row("1", "Escherichia coli", "active"),
        _label_row("2", "Escherichia coli", "inactive"),
        _label_row("3", "Escherichia coli", "uncertain"),
        _label_row("4", "Bacillus subtilis", "unlabeled"),
    ]
    result = filter_labels(rows, ["active", "inactive"])
    assert [r["peptide_id"] for r in result] == ["1", "2"]


def test_filter_labels_with_empty_included_list_drops_everything():
    rows = [_label_row("1", "Escherichia coli", "active")]
    assert filter_labels(rows, []) == []


def test_join_matches_by_peptide_id_and_organism():
    regression_rows = [
        _regression_row("1", "Escherichia coli", sequence="AAA", smiles="C1", mic_value_uM="10.0"),
        _regression_row("1", "Staphylococcus aureus", sequence="AAA", smiles="C1", mic_value_uM="50.0"),
    ]
    filtered_label_rows = [
        _label_row("1", "Escherichia coli", "active"),
        _label_row("1", "Staphylococcus aureus", "inactive"),
    ]
    result = join_regression_and_labels(regression_rows, filtered_label_rows)
    by_organism = {r["organism"]: r for r in result}
    assert by_organism["Escherichia coli"]["mic_value_uM"] == "10.0"
    assert by_organism["Escherichia coli"]["label"] == "active"
    assert by_organism["Staphylococcus aureus"]["mic_value_uM"] == "50.0"
    assert by_organism["Staphylococcus aureus"]["label"] == "inactive"


def test_join_raises_on_missing_regression_row():
    regression_rows = [_regression_row("1", "Escherichia coli")]
    filtered_label_rows = [_label_row("2", "Escherichia coli", "active")]
    with pytest.raises(DatasetAssemblyError):
        join_regression_and_labels(regression_rows, filtered_label_rows)


def test_attach_split_assigns_train_and_test():
    rows = [{"peptide_id": "1"}, {"peptide_id": "2"}]
    result = attach_split(rows, train_peptide_ids=["1"], test_peptide_ids=["2"])
    by_pid = {r["peptide_id"]: r["split"] for r in result}
    assert by_pid == {"1": "train", "2": "test"}


def test_attach_split_raises_on_peptide_id_with_no_assignment():
    rows = [{"peptide_id": "1"}]
    with pytest.raises(DatasetAssemblyError):
        attach_split(rows, train_peptide_ids=[], test_peptide_ids=[])


def test_attach_split_raises_on_peptide_id_in_both_sets():
    rows = [{"peptide_id": "1"}]
    with pytest.raises(DatasetAssemblyError):
        attach_split(rows, train_peptide_ids=["1"], test_peptide_ids=["1"])


def test_build_classification_dataset_end_to_end():
    regression_rows = [
        _regression_row("1", "Escherichia coli", mic_value_uM="10.0"),
        _regression_row("2", "Staphylococcus aureus", mic_value_uM="200.0"),
        _regression_row("3", "Escherichia coli", mic_value_uM="80.0"),
        _regression_row("4", "Bacillus subtilis", mic_value_uM="5.0"),
    ]
    label_rows = [
        _label_row("1", "Escherichia coli", "active"),
        _label_row("2", "Staphylococcus aureus", "inactive"),
        _label_row("3", "Escherichia coli", "uncertain"),
        _label_row("4", "Bacillus subtilis", "unlabeled"),
    ]
    result = build_classification_dataset(
        regression_rows, label_rows,
        train_peptide_ids=["1"], test_peptide_ids=["2"],
        included_labels=["active", "inactive"],
    )
    assert len(result) == 2
    organisms = {r["organism"] for r in result}
    assert organisms == {"Escherichia coli", "Staphylococcus aureus"}
    assert "Bacillus subtilis" not in organisms
    for row in result:
        assert set(row.keys()) == set(FIELDNAMES)
    by_pid = {r["peptide_id"]: r["split"] for r in result}
    assert by_pid == {"1": "train", "2": "test"}


def test_build_classification_dataset_output_column_order():
    regression_rows = [_regression_row("1", "Escherichia coli")]
    label_rows = [_label_row("1", "Escherichia coli", "active")]
    result = build_classification_dataset(
        regression_rows, label_rows,
        train_peptide_ids=["1"], test_peptide_ids=[],
        included_labels=["active", "inactive"],
    )
    assert list(result[0].keys()) == FIELDNAMES
