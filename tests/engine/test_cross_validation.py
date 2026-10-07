import csv

import numpy as np
import pandas as pd
import pytest

from soamp.engine.cross_validation import (
    CrossValidationError,
    assign_train_folds,
    load_peptide_split,
    summarize_cv_metrics,
    train_and_evaluate_fold,
)


def _write_split_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["peptide_id", "community", "split", "fold_id", "has_noncanonical"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_load_peptide_split_parses_train_folds_and_test(tmp_path):
    path = tmp_path / "split.csv"
    _write_split_csv(path, [
        {"peptide_id": "1", "community": "5", "split": "train", "fold_id": "0", "has_noncanonical": "False"},
        {"peptide_id": "2", "community": "6", "split": "train", "fold_id": "3.0", "has_noncanonical": "True"},
        {"peptide_id": "3", "community": "7", "split": "test", "fold_id": "", "has_noncanonical": "False"},
    ])
    assert load_peptide_split(path) == {
        "1": {"split": "train", "fold_id": "0"},
        "2": {"split": "train", "fold_id": "3"},
        "3": {"split": "test", "fold_id": None},
    }


def test_assign_train_folds_keeps_train_only_adds_fold_without_mutating_input():
    rows = [{"peptide_id": "p1"}, {"peptide_id": "p2"}, {"peptide_id": "p3"}]
    split = {
        "p1": {"split": "train", "fold_id": "0"},
        "p2": {"split": "train", "fold_id": "1"},
        "p3": {"split": "test", "fold_id": None},
    }
    result = assign_train_folds(rows, split)
    assert [(r["peptide_id"], r["fold_id"]) for r in result] == [("p1", "0"), ("p2", "1")]
    assert "fold_id" not in rows[0]  # original dicts untouched


def test_assign_train_folds_same_sequence_different_folds_keep_their_own_fold():
    # 'X' placeholder sequences are shared by different molecules; a
    # sequence-keyed lookup would collapse these onto one fold.
    rows = [{"peptide_id": "p1", "sequence": "AXA"}, {"peptide_id": "p2", "sequence": "AXA"}]
    split = {"p1": {"split": "train", "fold_id": "0"}, "p2": {"split": "train", "fold_id": "4"}}
    assert [r["fold_id"] for r in assign_train_folds(rows, split)] == ["0", "4"]


def test_assign_train_folds_raises_on_unmapped_peptide_or_missing_fold():
    with pytest.raises(CrossValidationError, match="ZZZ"):
        assign_train_folds([{"peptide_id": "ZZZ"}], {"p1": {"split": "train", "fold_id": "0"}})
    with pytest.raises(CrossValidationError, match="no fold_id"):
        assign_train_folds([{"peptide_id": "p1"}], {"p1": {"split": "train", "fold_id": None}})


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


def test_train_and_evaluate_fold_on_epoch_end_reports_finite_losses_per_epoch():
    rows = _make_peptide_rows(40)
    row_groups = {"fit": rows[:30], "val": rows[30:]}
    history = []

    train_and_evaluate_fold(
        row_groups, epochs=3, seed=42, peptide_method="rdkit_descriptors",
        organism_method="vocab_embedding", architecture="baseline_classifier",
        device="cpu", on_epoch_end=history.append,
    )
    assert [h["epoch"] for h in history] == [1, 2, 3]
    for h in history:
        assert set(h) == {"epoch", "fit_loss", "val_loss", "val_auroc"}
        assert np.isfinite([h["fit_loss"], h["val_loss"]]).all()


def test_train_and_evaluate_fold_without_callback_does_not_need_a_val_group():
    rows = _make_peptide_rows(20)
    metrics_by_group, _ = train_and_evaluate_fold(
        {"fit": rows}, eval_groups=("fit",), epochs=1, seed=42,
        peptide_method="rdkit_descriptors", organism_method="vocab_embedding",
        architecture="baseline_classifier", device="cpu",
    )
    assert set(metrics_by_group) == {"fit"}


def test_train_and_evaluate_fold_forwards_peptide_and_organism_method_kwargs():
    """This is the plumbing a caller relies on to share a PeptideCLMFeaturizer
    cache dict across folds (peptide_method_kwargs={"cache": shared_dict}) --
    verified here via a spy on build_dataset rather than exercising the real
    PeptideCLM model."""
    import soamp.engine.cross_validation as cv_module

    rows = _make_peptide_rows(20)
    row_groups = {"fit": rows[:15], "val": rows[15:]}
    peptide_kwargs = {"descriptor_names": ["MolWt", "TPSA"]}
    organism_kwargs = {"unknown_index": 0}

    original_build_dataset = cv_module.build_dataset
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original_build_dataset(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cv_module, "build_dataset", spy)
        train_and_evaluate_fold(
            row_groups,
            epochs=1,
            seed=42,
            peptide_method="rdkit_descriptors",
            peptide_method_kwargs=peptide_kwargs,
            organism_method="vocab_embedding",
            organism_method_kwargs=organism_kwargs,
            architecture="baseline_classifier",
            device="cpu",
        )

    assert len(calls) == 1
    assert calls[0]["peptide_method_kwargs"] == peptide_kwargs
    assert calls[0]["organism_method_kwargs"] == organism_kwargs


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
