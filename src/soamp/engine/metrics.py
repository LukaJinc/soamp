"""Pure metric computation from raw prediction arrays -- no model/dataset/
training-loop coupling.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class MetricsError(ValueError):
    """Raised when labels contains a single class (AUROC undefined)."""


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
    organism_idx: np.ndarray,
    index_to_organism: dict[int, str],
) -> dict[str, dict[str, float]]:
    """Buckets by organism_idx, compute_binary_metrics() per bucket, adds
    'n' (row count) per bucket. A bucket with a single label class present
    is skipped with a {'skipped': 'single_class', 'n': ...} entry rather
    than raising -- a degenerate per-organism slice shouldn't kill the
    whole eval."""
    result = {}
    for idx, name in index_to_organism.items():
        mask = organism_idx == idx
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
