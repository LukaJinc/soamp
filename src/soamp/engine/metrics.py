"""Pure metric computation from raw prediction arrays -- no model/dataset/
training-loop coupling.
"""
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsError(ValueError):
    """Raised when labels contains a single class (AUROC undefined), or when
    organism_names doesn't align row-for-row with labels."""


def logits_to_predictions(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs >= threshold).astype(int)


def prior_matched_predictions(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Predict "active" for the k highest-scoring rows, k = round(share of positives * n): a threshold
    chosen so the predicted-active rate equals the true rate. Removes the calibration shift that a
    class-weighted loss puts on the fixed 0.5 threshold, so F1/MCC measure ranking, not calibration."""
    n = len(labels)
    k = int(round(float(np.mean(labels)) * n))
    preds = np.zeros(n, dtype=int)
    preds[np.argsort(-logits, kind="stable")[:k]] = 1
    return preds


def compute_binary_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """{'accuracy', 'f1', 'precision', 'recall', 'auroc', 'mcc'} at the 0.5 threshold, plus
    'f1_matched' / 'mcc_matched' at a prior-matched threshold (see prior_matched_predictions) and
    'pred_active_rate' (share predicted active at 0.5; compare with the true rate). Raises MetricsError
    on single-class labels -- surfaced loudly rather than silently NaN'd.
    precision/recall use zero_division=0 (explicit, deterministic) rather
    than sklearn's warning-and-NaN default -- a model that predicts no
    positives at all has a well-defined precision of 0, not an undefined
    value that would otherwise need special-casing downstream."""
    if len(set(labels.tolist())) < 2:
        raise MetricsError("labels contains a single class, AUROC is undefined")
    preds = logits_to_predictions(logits)
    probs = 1.0 / (1.0 + np.exp(-logits))
    matched = prior_matched_predictions(logits, labels)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probs)),
        # Matthews correlation: defined as 0.0 by sklearn when predictions are
        # constant, so a degenerate all-one-class model reads 0, not NaN.
        "mcc": float(matthews_corrcoef(labels, preds)),
        "f1_matched": float(f1_score(labels, matched)),
        "mcc_matched": float(matthews_corrcoef(labels, matched)),
        "pred_active_rate": float(np.mean(preds)),
    }


def compute_metrics_by_organism(
    logits: np.ndarray,
    labels: np.ndarray,
    organism_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Buckets by organism *name*, compute_binary_metrics() per bucket, adds
    'n' (row count) per bucket. A bucket with a single label class present
    is skipped with a {'skipped': 'single_class', 'n': ...} entry rather
    than raising -- a degenerate per-organism slice shouldn't kill the
    whole eval.

    Takes names rather than the model's encoded organism input, so this works
    identically for an "index" organism featurization (vocab_embedding) and a
    "vector" one (kmer_composition), where there is no vocab to invert and the
    encoded input is a float matrix. Callers pass the organism column straight
    off the eval rows -- valid as long as the eval DataLoader is built with
    shuffle=False, so row order is preserved.
    """
    organism_names = np.asarray(organism_names)
    if len(organism_names) != len(labels):
        raise MetricsError(
            f"organism_names has {len(organism_names)} entries but there are "
            f"{len(labels)} labels -- they must align row-for-row"
        )
    result = {}
    for name in sorted(set(organism_names.tolist())):
        mask = organism_names == name
        n = int(mask.sum())
        if n == 0:
            continue
        bucket_logits, bucket_labels = logits[mask], labels[mask]
        try:
            metrics = compute_binary_metrics(bucket_logits, bucket_labels)
            metrics["n"] = n
        except MetricsError:
            metrics = {"skipped": "single_class", "n": n}
        result[name] = metrics
    return result
