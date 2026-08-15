import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from soamp.engine.metrics import MetricsError, compute_binary_metrics, compute_metrics_by_organism


def test_compute_binary_metrics_matches_sklearn_directly():
    logits = np.array([2.0, -2.0, 1.0, -1.0, 3.0])
    labels = np.array([1, 0, 1, 1, 0])
    result = compute_binary_metrics(logits, labels)

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    assert result["accuracy"] == pytest.approx(accuracy_score(labels, preds))
    assert result["f1"] == pytest.approx(f1_score(labels, preds))
    assert result["auroc"] == pytest.approx(roc_auc_score(labels, probs))


def test_compute_binary_metrics_raises_on_single_class_labels():
    logits = np.array([1.0, 2.0, 3.0])
    labels = np.array([1, 1, 1])
    with pytest.raises(MetricsError):
        compute_binary_metrics(logits, labels)


def test_compute_metrics_by_organism_buckets_correctly():
    logits = np.array([2.0, -2.0, 1.0, -1.0])
    labels = np.array([1, 0, 1, 0])
    organism_idx = np.array([1, 1, 2, 2])
    result = compute_metrics_by_organism(
        logits, labels, organism_idx, {1: "Escherichia coli", 2: "Staphylococcus aureus"}
    )
    assert set(result.keys()) == {"Escherichia coli", "Staphylococcus aureus"}
    assert result["Escherichia coli"]["n"] == 2
    assert result["Staphylococcus aureus"]["n"] == 2


def test_compute_metrics_by_organism_skips_degenerate_bucket_without_raising():
    logits = np.array([2.0, -2.0, 1.0])
    labels = np.array([1, 0, 1])  # organism 2 bucket is single-class
    organism_idx = np.array([1, 1, 2])
    result = compute_metrics_by_organism(logits, labels, organism_idx, {1: "org_a", 2: "org_b"})
    assert result["org_b"]["skipped"] == "single_class"
    assert result["org_b"]["n"] == 1
    assert "accuracy" in result["org_a"]


def test_compute_metrics_by_organism_skips_absent_organism():
    logits = np.array([2.0, -2.0])
    labels = np.array([1, 0])
    organism_idx = np.array([1, 1])
    result = compute_metrics_by_organism(logits, labels, organism_idx, {1: "org_a", 2: "org_b"})
    assert "org_b" not in result
