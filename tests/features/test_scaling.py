import pytest

from soamp.features.scaling import ScalingError, apply_scaler, fit_scaler


def test_fit_scaler_mean_scale_length_matches_descriptors():
    rows = [{"a": 1.0, "b": 10.0}, {"a": 3.0, "b": 30.0}]
    result = fit_scaler(rows, ["a", "b"])
    assert len(result["mean"]) == 2
    assert len(result["scale"]) == 2
    assert result["descriptor_names"] == ["a", "b"]


def test_fit_scaler_raises_on_empty_rows():
    with pytest.raises(ScalingError):
        fit_scaler([], ["a"])


def test_fit_scaler_raises_on_missing_descriptor():
    rows = [{"a": 1.0}]
    with pytest.raises(ScalingError):
        fit_scaler(rows, ["a", "b"])


def test_apply_scaler_zero_mean_unit_variance_on_fit_data():
    rows = [{"a": 1.0}, {"a": 3.0}]
    fitted = fit_scaler(rows, ["a"])
    scaled = [apply_scaler([r["a"]], fitted["mean"], fitted["scale"])[0] for r in rows]
    assert sum(scaled) == pytest.approx(0.0, abs=1e-9)


def test_apply_scaler_raises_on_length_mismatch():
    with pytest.raises(ScalingError):
        apply_scaler([1.0, 2.0], [0.0], [1.0])
