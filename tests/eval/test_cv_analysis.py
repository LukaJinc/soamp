import numpy as np
import pandas as pd
import pytest

from soamp.eval.cv_analysis import (
    CVAnalysisError,
    noncanonical_recall,
    noncanonical_recall_table,
    per_fold_fit_val,
    pretty_label,
)


def _rows(group, fold, labels, preds, noncanonical=None):
    """logit +5 -> predicted active, -5 -> predicted inactive."""
    noncanonical = noncanonical if noncanonical is not None else [False] * len(labels)
    return pd.DataFrame({
        "eval_group": group, "val_fold_id": fold,
        "true_label_int": labels,
        "logit": [5.0 if p else -5.0 for p in preds],
        "has_noncanonical": noncanonical,
    })


def _two_fold_frame():
    return pd.concat([
        # fold 0 val: TP=2 FN=1 FP=0 TN=1 -> precision 1.0, recall 2/3
        _rows("val", 0, [1, 1, 1, 0], [1, 1, 0, 0]),
        # fold 0 fit: TP=1 FN=1 FP=1 TN=1 -> precision .5, recall .5
        _rows("fit", 0, [1, 1, 0, 0], [1, 0, 1, 0]),
        # fold 1 val: perfect
        _rows("val", 1, [1, 0, 1, 0], [1, 0, 1, 0]),
        _rows("fit", 1, [1, 0, 1, 0], [1, 0, 1, 0]),
    ], ignore_index=True)


def test_per_fold_fit_val_known_precision_and_recall():
    out = per_fold_fit_val(_two_fold_frame())
    assert list(out.columns[:2]) == ["fold", "eval_group"]
    val0 = out[(out.fold == 0) & (out.eval_group == "val")].iloc[0]
    fit0 = out[(out.fold == 0) & (out.eval_group == "fit")].iloc[0]
    assert val0.precision == pytest.approx(1.0) and val0.recall == pytest.approx(2 / 3)
    assert fit0.precision == pytest.approx(0.5) and fit0.recall == pytest.approx(0.5)
    assert len(out) == 4 and set(out.eval_group) == {"fit", "val"}


def test_per_fold_fit_val_rejects_missing_columns():
    with pytest.raises(CVAnalysisError, match="has_noncanonical"):
        per_fold_fit_val(_two_fold_frame().drop(columns="has_noncanonical"))


def _nc_frame():
    canonical = _rows("val", 0, [1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 1], [False] * 6)   # recall 3/4
    noncanonical = _rows("val", 1, [1, 1, 0, 0], [1, 0, 0, 0], [True] * 4)             # recall 1/2
    train_rows = _rows("fit", 0, [1, 1, 0, 0], [0, 0, 1, 1], [True] * 4)               # must be ignored
    return pd.concat([canonical, noncanonical, train_rows], ignore_index=True)


def test_noncanonical_recall_uses_only_val_rows_and_reports_delta_and_counts():
    out = noncanonical_recall(_nc_frame())
    assert out["canonical_recall"] == pytest.approx(0.75)
    assert out["noncanonical_recall"] == pytest.approx(0.5)
    assert out["delta"] == pytest.approx(-0.25)
    assert (out["n_canonical"], out["n_noncanonical"]) == (6, 4)


def test_noncanonical_recall_accepts_string_booleans_from_csv_roundtrip():
    frame = _nc_frame()
    frame["has_noncanonical"] = frame["has_noncanonical"].map({True: "True", False: "False"})
    # a CSV/wandb round trip can hand back strings; "False" must not become truthy
    frame["has_noncanonical"] = frame["has_noncanonical"] == "True"
    assert noncanonical_recall(frame)["noncanonical_recall"] == pytest.approx(0.5)


def test_noncanonical_recall_raises_without_a_noncanonical_slice():
    only_canonical = _rows("val", 0, [1, 0, 1, 0], [1, 0, 1, 0], [False] * 4)
    with pytest.raises(CVAnalysisError, match="noncanonical"):
        noncanonical_recall(only_canonical)


def test_noncanonical_recall_table_keeps_run_order():
    table = noncanonical_recall_table({"b_run": _nc_frame(), "a_run": _nc_frame()})
    assert list(table.index) == ["b_run", "a_run"]
    assert np.allclose(table["delta"], -0.25)


@pytest.mark.parametrize("exp_id, expected", [
    ("rdkit_vocab_cv", "RDKit × vocab"),
    ("peptideclm_kmer_cv", "PeptideCLM × k-mer"),
    ("molgraph_dnabert_cv", "GNN × DNABERT-S"),
    ("something_else", "something_else"),
])
def test_pretty_label(exp_id, expected):
    assert pretty_label(exp_id) == expected
