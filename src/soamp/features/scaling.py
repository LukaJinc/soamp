"""Pure StandardScaler fit/apply for the peptide descriptor vector.

Fit is sklearn-backed (for correctness/parity with sklearn's variance
convention); apply is plain arithmetic so consumers (the torch Dataset)
don't need sklearn at __getitem__ time.
"""
from sklearn.preprocessing import StandardScaler


class ScalingError(ValueError):
    """Raised when feature_rows is empty, a row is missing one of
    descriptor_names, or values/mean/scale lengths disagree."""


def fit_scaler(feature_rows: list[dict], descriptor_names: list[str]) -> dict:
    """Fits a StandardScaler on feature_rows[*][name for name in
    descriptor_names]. Returns {"descriptor_names": [...], "mean": [...],
    "scale": [...]} (plain lists, JSON-serializable -- no pickled sklearn
    object, so the scaler artifact isn't coupled to a specific sklearn
    version)."""
    if not feature_rows:
        raise ScalingError("feature_rows is empty")
    try:
        matrix = [[row[name] for name in descriptor_names] for row in feature_rows]
    except KeyError as e:
        raise ScalingError(f"feature row missing descriptor {e}") from e
    scaler = StandardScaler()
    scaler.fit(matrix)
    return {
        "descriptor_names": list(descriptor_names),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }


def apply_scaler(values: list[float], mean: list[float], scale: list[float]) -> list[float]:
    """(x - mean) / scale, elementwise. Raises ScalingError on length
    mismatch."""
    if not (len(values) == len(mean) == len(scale)):
        raise ScalingError(
            f"length mismatch: values={len(values)}, mean={len(mean)}, scale={len(scale)}"
        )
    return [(v - m) / s for v, m, s in zip(values, mean, scale)]
