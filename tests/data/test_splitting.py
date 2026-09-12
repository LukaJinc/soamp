import pytest

from soamp.data.splitting import SplitError, apply_val_split, split_train_validation

# 5 pairs of peptides; each pair is a single amino-acid substitution apart
# (>=90% identity), so at threshold=0.6 the two members of a pair must
# always cluster -- and therefore split -- together.
_PAIRS = [
    ("1", "AAAAAAAAAA"), ("2", "AAAAAAAAAC"),
    ("3", "KKKKKKKKKK"), ("4", "KKKKKKKKKR"),
    ("5", "WWWWWWWWWW"), ("6", "WWWWWWWWWY"),
    ("7", "FFFFFFFFFF"), ("8", "FFFFFFFFFL"),
    ("9", "GGGGGGGGGG"), ("10", "GGGGGGGGGA"),
]


def _rows():
    rows = [{"peptide_id": pid, "organism": "Escherichia coli", "sequence": seq}
            for pid, seq in _PAIRS]
    # peptide_id "1" also has a second organism's row -- must land in one bucket.
    rows.append({"peptide_id": "1", "organism": "Staphylococcus aureus", "sequence": "AAAAAAAAAA"})
    return rows


def test_split_keeps_all_rows_of_a_peptide_id_together():
    fit_rows, val_rows = split_train_validation(_rows(), val_fraction=0.3, seed=42)
    fit_ids = {r["peptide_id"] for r in fit_rows}
    val_ids = {r["peptide_id"] for r in val_rows}
    assert fit_ids.isdisjoint(val_ids)
    # peptide_id "1" has 2 rows -- both must be on the same side.
    id1_rows = [r for r in (fit_rows + val_rows) if r["peptide_id"] == "1"]
    assert len(id1_rows) == 2


def test_split_is_disjoint_and_covers_all_rows():
    rows = _rows()
    fit_rows, val_rows = split_train_validation(rows, val_fraction=0.3, seed=42)
    assert len(fit_rows) + len(val_rows) == len(rows)


def test_split_keeps_near_identical_pairs_together():
    # Each pair in _PAIRS is a single-substitution near-duplicate -- a
    # homology-aware split must never put the two halves of a pair on
    # opposite sides, unlike a naive random-by-peptide_id shuffle.
    fit_rows, val_rows = split_train_validation(_rows(), val_fraction=0.3, seed=42)
    fit_ids = {r["peptide_id"] for r in fit_rows}
    val_ids = {r["peptide_id"] for r in val_rows}
    for pid_a, pid_b in [("1", "2"), ("3", "4"), ("5", "6"), ("7", "8"), ("9", "10")]:
        same_side = (pid_a in fit_ids and pid_b in fit_ids) or (pid_a in val_ids and pid_b in val_ids)
        assert same_side, f"pair ({pid_a}, {pid_b}) split across fit/val"


def test_split_is_deterministic_given_seed():
    rows = _rows()
    fit1, val1 = split_train_validation(rows, val_fraction=0.3, seed=42)
    fit2, val2 = split_train_validation(rows, val_fraction=0.3, seed=42)
    assert {r["peptide_id"] for r in val1} == {r["peptide_id"] for r in val2}


def test_split_raises_on_bad_fraction():
    with pytest.raises(SplitError):
        split_train_validation(_rows(), val_fraction=1.0, seed=42)
    with pytest.raises(SplitError):
        split_train_validation(_rows(), val_fraction=0.0, seed=42)


def test_split_raises_when_fraction_rounds_to_zero():
    rows = [{"peptide_id": "1", "organism": "Escherichia coli", "sequence": "AAAAAAAAAA"}]
    with pytest.raises(SplitError):
        split_train_validation(rows, val_fraction=0.1, seed=42)


def test_apply_val_split_partitions_by_peptide_id():
    train_rows = [
        {"peptide_id": "1", "organism": "Escherichia coli"},
        {"peptide_id": "2", "organism": "Staphylococcus aureus"},
        {"peptide_id": "3", "organism": "Pseudomonas aeruginosa"},
    ]
    fit_rows, val_rows = apply_val_split(train_rows, val_peptide_ids={"2"})
    assert {r["peptide_id"] for r in fit_rows} == {"1", "3"}
    assert {r["peptide_id"] for r in val_rows} == {"2"}


def test_apply_val_split_empty_val_ids_puts_everything_in_fit():
    train_rows = [{"peptide_id": "1", "organism": "Escherichia coli"}]
    fit_rows, val_rows = apply_val_split(train_rows, val_peptide_ids=set())
    assert fit_rows == train_rows
    assert val_rows == []
