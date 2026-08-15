import pytest

from soamp.data.splitting import SplitError, split_train_validation


def _rows():
    # peptide_id "1" appears twice (2 organisms) -- must land in one bucket.
    return [
        {"peptide_id": "1", "organism": "Escherichia coli"},
        {"peptide_id": "1", "organism": "Staphylococcus aureus"},
        {"peptide_id": "2", "organism": "Escherichia coli"},
        {"peptide_id": "3", "organism": "Escherichia coli"},
        {"peptide_id": "4", "organism": "Escherichia coli"},
        {"peptide_id": "5", "organism": "Escherichia coli"},
        {"peptide_id": "6", "organism": "Escherichia coli"},
        {"peptide_id": "7", "organism": "Escherichia coli"},
        {"peptide_id": "8", "organism": "Escherichia coli"},
        {"peptide_id": "9", "organism": "Escherichia coli"},
        {"peptide_id": "10", "organism": "Escherichia coli"},
    ]


def test_split_keeps_all_rows_of_a_peptide_id_together():
    fit_rows, val_rows = split_train_validation(_rows(), val_fraction=0.2, seed=42)
    fit_ids = {r["peptide_id"] for r in fit_rows}
    val_ids = {r["peptide_id"] for r in val_rows}
    assert fit_ids.isdisjoint(val_ids)
    # peptide_id "1" has 2 rows -- both must be on the same side.
    id1_rows = [r for r in (fit_rows + val_rows) if r["peptide_id"] == "1"]
    assert len(id1_rows) == 2


def test_split_is_disjoint_and_covers_all_rows():
    rows = _rows()
    fit_rows, val_rows = split_train_validation(rows, val_fraction=0.2, seed=42)
    assert len(fit_rows) + len(val_rows) == len(rows)


def test_split_respects_fraction_approximately():
    rows = _rows()
    fit_rows, val_rows = split_train_validation(rows, val_fraction=0.2, seed=42)
    n_val_ids = len({r["peptide_id"] for r in val_rows})
    n_total_ids = len({r["peptide_id"] for r in rows})
    assert n_val_ids == round(n_total_ids * 0.2)


def test_split_is_deterministic_given_seed():
    rows = _rows()
    fit1, val1 = split_train_validation(rows, val_fraction=0.3, seed=7)
    fit2, val2 = split_train_validation(rows, val_fraction=0.3, seed=7)
    assert {r["peptide_id"] for r in val1} == {r["peptide_id"] for r in val2}


def test_split_raises_on_bad_fraction():
    with pytest.raises(SplitError):
        split_train_validation(_rows(), val_fraction=1.0, seed=42)
    with pytest.raises(SplitError):
        split_train_validation(_rows(), val_fraction=0.0, seed=42)


def test_split_raises_when_fraction_rounds_to_zero():
    rows = [{"peptide_id": "1", "organism": "Escherichia coli"}]
    with pytest.raises(SplitError):
        split_train_validation(rows, val_fraction=0.1, seed=42)
