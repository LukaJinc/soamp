import pytest

from soamp.labeling.labels import derive_label


def test_no_threshold_is_unlabeled():
    label, reason = derive_label(10.0, "exact", None, None)
    assert label == "unlabeled"
    assert reason == "no_threshold_available"


@pytest.mark.parametrize("mic_type", ["exact", "averaged"])
def test_direct_comparison_active(mic_type):
    label, reason = derive_label(10.0, mic_type, active_threshold_uM=32.0, inactive_threshold_uM=32.0)
    assert label == "active"
    assert reason == "value_at_or_below_active_threshold"


@pytest.mark.parametrize("mic_type", ["exact", "averaged"])
def test_direct_comparison_inactive(mic_type):
    label, reason = derive_label(50.0, mic_type, active_threshold_uM=32.0, inactive_threshold_uM=32.0)
    assert label == "inactive"
    assert reason == "value_at_or_above_inactive_threshold"


def test_direct_comparison_boundary_is_active():
    # mic_value_uM == active_threshold_uM: "active if <= active_threshold"
    label, _ = derive_label(32.0, "exact", active_threshold_uM=32.0, inactive_threshold_uM=32.0)
    assert label == "active"


def test_direct_comparison_gray_zone_is_uncertain():
    # strictly between the two breakpoints -- neither provably active nor inactive
    label, reason = derive_label(64.0, "exact", active_threshold_uM=32.0, inactive_threshold_uM=128.0)
    assert label == "uncertain"
    assert reason == "value_in_gray_zone_or_straddles_threshold"


def test_direct_comparison_at_inactive_breakpoint_is_inactive():
    label, _ = derive_label(128.0, "exact", active_threshold_uM=32.0, inactive_threshold_uM=128.0)
    assert label == "inactive"


def test_left_censored_upper_bound_below_threshold_is_active():
    # true value is in [0, 0.5] -- provably active for any active threshold >= 0.5
    label, reason = derive_label(
        0.5, "censored", active_threshold_uM=32.0, inactive_threshold_uM=32.0,
        recovered_min_uM=0.0, recovered_max_uM=0.5,
    )
    assert label == "active"
    assert reason == "value_at_or_below_active_threshold"


def test_left_censored_upper_bound_above_threshold_is_uncertain():
    # true value is in [0, 50] -- could be above or below a threshold of 32
    label, reason = derive_label(
        50.0, "censored", active_threshold_uM=32.0, inactive_threshold_uM=32.0,
        recovered_min_uM=0.0, recovered_max_uM=50.0,
    )
    assert label == "uncertain"


def test_right_censored_lower_bound_above_threshold_is_inactive():
    # true value is in [100, inf) -- provably inactive for any inactive threshold <= 100
    label, reason = derive_label(
        100.0, "censored", active_threshold_uM=32.0, inactive_threshold_uM=32.0,
        recovered_min_uM=100.0, recovered_max_uM=float("inf"),
    )
    assert label == "inactive"
    assert reason == "value_at_or_above_inactive_threshold"


def test_right_censored_lower_bound_below_threshold_is_uncertain():
    # true value is in [20, inf) -- could be above or below a threshold of 32
    label, reason = derive_label(
        20.0, "censored", active_threshold_uM=32.0, inactive_threshold_uM=32.0,
        recovered_min_uM=20.0, recovered_max_uM=float("inf"),
    )
    assert label == "uncertain"


def test_censored_interval_inside_gray_zone_is_uncertain():
    # true value is in [40, 60], entirely inside the (32, 128) gray zone
    label, reason = derive_label(
        50.0, "censored", active_threshold_uM=32.0, inactive_threshold_uM=128.0,
        recovered_min_uM=40.0, recovered_max_uM=60.0,
    )
    assert label == "uncertain"


def test_two_sided_range_fully_below_active_threshold_is_active():
    label, reason = derive_label(
        7.0, "censored", active_threshold_uM=32.0, inactive_threshold_uM=32.0,
        recovered_min_uM=4.0, recovered_max_uM=10.0,
    )
    assert label == "active"
    assert reason == "value_at_or_below_active_threshold"


def test_two_sided_range_straddling_threshold_is_uncertain():
    label, reason = derive_label(
        30.0, "censored", active_threshold_uM=32.0, inactive_threshold_uM=32.0,
        recovered_min_uM=20.0, recovered_max_uM=40.0,
    )
    assert label == "uncertain"


def test_censored_without_bounds_raises():
    with pytest.raises(ValueError):
        derive_label(10.0, "censored", active_threshold_uM=32.0, inactive_threshold_uM=32.0)


def test_unknown_mic_type_raises():
    with pytest.raises(ValueError):
        derive_label(10.0, "bogus", active_threshold_uM=32.0, inactive_threshold_uM=32.0)
