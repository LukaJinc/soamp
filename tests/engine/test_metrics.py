import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from soamp.engine.metrics import MetricsError, compute_binary_metrics, compute_metrics_by_organism


def test_compute_binary_metrics_matches_sklearn_directly():
    logits = np.array([2.0, -2.0, 1.0, -1.0, 3.0])
    labels = np.array([1, 0, 1, 1, 0])
    result = compute_binary_metrics(logits, labels)

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    assert result["accuracy"] == pytest.approx(accuracy_score(labels, preds))
    assert result["f1"] == pytest.approx(f1_score(labels, preds))
    assert result["precision"] == pytest.approx(precision_score(labels, preds, zero_division=0))
    assert result["recall"] == pytest.approx(recall_score(labels, preds, zero_division=0))
    assert result["auroc"] == pytest.approx(roc_auc_score(labels, probs))


def test_compute_binary_metrics_precision_recall_are_well_defined_when_no_positives_predicted():
    """A model that predicts every row negative has a well-defined precision
    of 0 (zero_division=0), not sklearn's default warning-and-NaN -- the
    labels themselves still have both classes present, so this isn't the
    MetricsError case."""
    logits = np.array([-5.0, -5.0, -5.0])
    labels = np.array([1, 0, 0])
    result = compute_binary_metrics(logits, labels)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_compute_binary_metrics_precision_recall_perfect_predictions():
    logits = np.array([5.0, -5.0, 5.0, -5.0])
    labels = np.array([1, 0, 1, 0])
    result = compute_binary_metrics(logits, labels)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_compute_binary_metrics_raises_on_single_class_labels():
    logits = np.array([1.0, 2.0, 3.0])
    labels = np.array([1, 1, 1])
    with pytest.raises(MetricsError):
        compute_binary_metrics(logits, labels)


def test_compute_metrics_by_organism_buckets_correctly():
    logits = np.array([2.0, -2.0, 1.0, -1.0])
    labels = np.array([1, 0, 1, 0])
    organism_names = ["Escherichia coli", "Escherichia coli",
                      "Staphylococcus aureus", "Staphylococcus aureus"]
    result = compute_metrics_by_organism(logits, labels, organism_names)
    assert set(result.keys()) == {"Escherichia coli", "Staphylococcus aureus"}
    assert result["Escherichia coli"]["n"] == 2
    assert result["Staphylococcus aureus"]["n"] == 2


def test_compute_metrics_by_organism_skips_degenerate_bucket_without_raising():
    logits = np.array([2.0, -2.0, 1.0])
    labels = np.array([1, 0, 1])  # org_b bucket is single-class
    result = compute_metrics_by_organism(logits, labels, ["org_a", "org_a", "org_b"])
    assert result["org_b"]["skipped"] == "single_class"
    assert result["org_b"]["n"] == 1
    assert "accuracy" in result["org_a"]


def test_compute_metrics_by_organism_only_reports_organisms_present():
    logits = np.array([2.0, -2.0])
    labels = np.array([1, 0])
    result = compute_metrics_by_organism(logits, labels, ["org_a", "org_a"])
    assert set(result.keys()) == {"org_a"}


def test_compute_metrics_by_organism_works_for_vector_organism_featurization():
    """The whole point of bucketing by name: a "vector" organism strategy
    (kmer_composition) has no vocab to invert, and its encoded model input is
    an (N, D) float matrix rather than an index per row. Names come off the
    eval rows, so nothing here depends on how the organism was encoded."""
    logits = np.array([2.0, -2.0, 1.0, -1.0])
    labels = np.array([1, 0, 1, 0])
    names = ["Escherichia coli", "Escherichia coli",
             "Pseudomonas aeruginosa", "Pseudomonas aeruginosa"]
    result = compute_metrics_by_organism(logits, labels, names)
    assert set(result) == {"Escherichia coli", "Pseudomonas aeruginosa"}
    assert all(m["n"] == 2 for m in result.values())


def test_compute_metrics_by_organism_raises_on_length_mismatch():
    logits = np.array([2.0, -2.0, 1.0])
    labels = np.array([1, 0, 1])
    with pytest.raises(MetricsError, match="align row-for-row"):
        compute_metrics_by_organism(logits, labels, ["org_a", "org_a"])


def test_compute_binary_metrics_includes_mcc_matching_sklearn():
    from sklearn.metrics import matthews_corrcoef

    logits = np.array([2.0, -1.0, 0.5, -3.0, 1.5, -0.2])
    labels = np.array([1, 0, 0, 0, 1, 1])
    preds = (logits >= 0).astype(int)
    m = compute_binary_metrics(logits, labels)
    assert m["mcc"] == pytest.approx(matthews_corrcoef(labels, preds))


def test_mcc_is_zero_not_nan_for_constant_predictions():
    m = compute_binary_metrics(np.array([5.0, 5.0, 5.0, 5.0]), np.array([1, 0, 1, 0]))
    assert m["mcc"] == 0.0


def test_prior_matched_predictions_predict_the_true_rate():
    from soamp.engine.metrics import prior_matched_predictions

    logits = np.array([3.0, 2.0, 1.0, 0.5, -1.0, -2.0, -3.0, -4.0])
    labels = np.array([1, 1, 1, 0, 1, 0, 0, 0])          # 4 of 8 active
    preds = prior_matched_predictions(logits, labels)
    assert preds.sum() == 4 and preds[:4].tolist() == [1, 1, 1, 1]


def test_matched_metrics_ignore_a_constant_logit_shift_but_the_fixed_threshold_does_not():
    rng = np.random.default_rng(3)
    labels = (rng.random(400) < 0.85).astype(int)
    logits = labels * 1.2 + rng.normal(size=400) - 0.2
    base = compute_binary_metrics(logits, labels)
    shifted = compute_binary_metrics(logits - 2.0, labels)   # same ranking, calibration shifted down
    assert shifted["f1_matched"] == pytest.approx(base["f1_matched"])
    assert shifted["mcc_matched"] == pytest.approx(base["mcc_matched"])
    assert shifted["pred_active_rate"] < base["pred_active_rate"] and shifted["f1"] < base["f1"]
