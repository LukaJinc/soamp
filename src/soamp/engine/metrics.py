"""Pure metric computation from raw prediction arrays -- no model/dataset/
training-loop coupling.
"""
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class MetricsError(ValueError):
    """Raised when labels contains a single class (AUROC undefined), or when
    organism_names doesn't align row-for-row with labels."""


def logits_to_predictions(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs >= threshold).astype(int)


def compute_binary_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """{'accuracy', 'f1', 'auroc'}. Raises MetricsError on single-class
    labels -- surfaced loudly rather than silently NaN'd."""
    if len(set(labels.tolist())) < 2:
        raise MetricsError("labels contains a single class, AUROC is undefined")
    preds = logits_to_predictions(logits)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "auroc": float(roc_auc_score(labels, probs)),
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
