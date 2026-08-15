"""Pure class-balancing helpers for BCEWithLogitsLoss's pos_weight."""


class ClassBalancingError(ValueError):
    """Raised when either class is absent (pos_weight undefined), or an
    unknown mode is requested."""


def compute_pos_weight(labels: list[int]) -> float:
    """pos_weight = n_negative / n_positive -- the standard
    BCEWithLogitsLoss formula; up/down-weights the label=1 class's loss
    contribution to match the label=0 class's prevalence. Raises
    ClassBalancingError if either class has zero members."""
    n_pos = sum(1 for label in labels if label == 1)
    n_neg = sum(1 for label in labels if label == 0)
    if n_pos == 0 or n_neg == 0:
        raise ClassBalancingError(
            f"pos_weight undefined: n_positive={n_pos}, n_negative={n_neg}"
        )
    return n_neg / n_pos


def resolve_pos_weight(
    mode: str, fixed_pos_weight: float | None, train_labels: list[int]
) -> float | None:
    """mode='auto' -> compute_pos_weight(train_labels); 'fixed' ->
    fixed_pos_weight; 'none' -> None (unweighted BCE). Raises
    ClassBalancingError on an unknown mode."""
    if mode == "auto":
        return compute_pos_weight(train_labels)
    if mode == "fixed":
        return fixed_pos_weight
    if mode == "none":
        return None
    raise ClassBalancingError(f"unknown class-balancing mode: {mode!r}")
