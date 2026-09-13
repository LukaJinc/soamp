import csv

import numpy as np
import pandas as pd
import pytest

from soamp.engine.cross_validation import (
    CrossValidationError,
    assign_fold_ids,
    load_fold_assignments,
    summarize_cv_metrics,
    train_and_evaluate_fold,
)


def _write_folds_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "community", "sequence", "fold_id"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_fold_assignments_keys_by_sequence_not_node_id(tmp_path):
    """The whole point of this module: node_id is an igraph vertex index,
    not this dataset's peptide_id -- sequence is the only sound join key."""
    path = tmp_path / "folds.csv"
    _write_folds_csv(path, [
        {"node_id": "0", "community": "5", "sequence": "AAA", "fold_id": "0"},
        {"node_id": "1", "community": "5", "sequence": "BBB", "fold_id": "1"},
    ])
    result = load_fold_assignments(path)
    assert result == {"AAA": "0", "BBB": "1"}


def test_assign_fold_ids_adds_fold_id_without_mutating_input():
    rows = [{"peptide_id": "p1", "sequence": "AAA"}, {"peptide_id": "p2", "sequence": "BBB"}]
    fold_by_sequence = {"AAA": "0", "BBB": "1"}
    result = assign_fold_ids(rows, fold_by_sequence)
    assert [r["fold_id"] for r in result] == ["0", "1"]
    assert "fold_id" not in rows[0]  # original dicts untouched


def test_assign_fold_ids_raises_on_unmapped_sequence():
    rows = [{"peptide_id": "p1", "sequence": "AAA"}, {"peptide_id": "p2", "sequence": "ZZZ"}]
    fold_by_sequence = {"AAA": "0"}
    with pytest.raises(CrossValidationError, match="ZZZ"):
        assign_fold_ids(rows, fold_by_sequence)


def _make_peptide_rows(n, seed=0, organism="Escherichia coli"):
    """Tiny synthetic dataset in the same row shape as
    mic_classification_dataset.csv -- reuses simple, valid SMILES so
    rdkit_descriptors featurization succeeds, matching the style of
    tests/data/test_factory.py's synthetic fixtures."""
    rng = np.random.default_rng(seed)
    smiles_pool = ["CC(=O)O", "CCO", "CCN", "CCC", "CCCl", "CCF", "CCBr", "CCI"]
    rows = []
    for i in range(n):
        rows.append({
            "peptide_id": f"pep_{i}",
            "sequence": f"SEQ{i}",
            "smiles": smiles_pool[i % len(smiles_pool)],
            "organism": organism,
            "has_noncanonical": "False",
            "label": "active" if rng.random() > 0.5 else "inactive",
        })
    return rows


def test_train_and_evaluate_fold_returns_expected_columns_and_row_count():
    rows = _make_peptide_rows(40)
    row_groups = {"fit": rows[:30], "val": rows[30:]}

    metrics_by_group, results_df = train_and_evaluate_fold(
        row_groups,
        epochs=1,
        seed=42,
        peptide_method="rdkit_descriptors",
        organism_method="vocab_embedding",
        architecture="baseline_classifier",
        device="cpu",
    )

    assert set(metrics_by_group.keys()) == {"fit", "val"}
    for group_metrics in metrics_by_group.values():
        assert set(group_metrics) == {"accuracy", "f1", "precision", "recall", "auroc"}

    expected_cols = {
        "peptide_id", "sequence", "smiles", "organism", "has_noncanonical", "label",
        "true_label_int", "pred_label_int", "logit", "prob", "correct", "eval_group",
    }
    assert expected_cols.issubset(set(results_df.columns))
    # one row per (input row, eval_group) -- "fit" rows appear once for the
    # fit-group evaluation, "val" rows once for the val-group evaluation.
    assert len(results_df) == 40
    assert results_df["eval_group"].value_counts().to_dict() == {"fit": 30, "val": 10}


def test_train_and_evaluate_fold_only_evaluates_requested_groups():
    rows = _make_peptide_rows(30)
    row_groups = {"fit": rows[:20], "val": rows[20:]}

    metrics_by_group, results_df = train_and_evaluate_fold(
        row_groups,
        eval_groups=("fit",),
        epochs=1,
        seed=42,
        peptide_method="rdkit_descriptors",
        organism_method="vocab_embedding",
        architecture="baseline_classifier",
        device="cpu",
    )
    assert set(metrics_by_group.keys()) == {"fit"}
    assert len(results_df) == 20
    assert set(results_df["eval_group"]) == {"fit"}


def test_summarize_cv_metrics_computes_mean_and_std_per_eval_group():
    fold_metrics_rows = [
        {"val_fold_id": 0, "eval_group": "fit", "accuracy": 0.8, "f1": 0.7},
        {"val_fold_id": 1, "eval_group": "fit", "accuracy": 0.82, "f1": 0.72},
        {"val_fold_id": 0, "eval_group": "val", "accuracy": 0.6, "f1": 0.5},
        {"val_fold_id": 1, "eval_group": "val", "accuracy": 0.58, "f1": 0.48},
    ]
    df = pd.DataFrame(fold_metrics_rows)
    summary = summarize_cv_metrics(df, ["accuracy", "f1"])

    assert summary.loc["fit", ("accuracy", "mean")] == pytest.approx(0.81)
    assert summary.loc["val", ("accuracy", "mean")] == pytest.approx(0.59)
    assert summary.loc["fit", ("f1", "mean")] == pytest.approx(0.71)
