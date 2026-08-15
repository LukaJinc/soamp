import pytest

from soamp.engine.class_balancing import ClassBalancingError, compute_pos_weight, resolve_pos_weight


def test_compute_pos_weight_matches_manual_formula():
    labels = [1, 1, 1, 1, 0]  # 4 positive, 1 negative
    assert compute_pos_weight(labels) == pytest.approx(1 / 4)


def test_compute_pos_weight_raises_when_one_class_absent():
    with pytest.raises(ClassBalancingError):
        compute_pos_weight([1, 1, 1])
    with pytest.raises(ClassBalancingError):
        compute_pos_weight([0, 0, 0])


@pytest.mark.parametrize(
    "mode,fixed,expected",
    [
        ("none", None, None),
        ("fixed", 2.5, 2.5),
    ],
)
def test_resolve_pos_weight_modes(mode, fixed, expected):
    assert resolve_pos_weight(mode, fixed, [1, 0]) == expected


def test_resolve_pos_weight_auto_mode_computes_from_labels():
    labels = [1, 1, 1, 1, 0]
    assert resolve_pos_weight("auto", None, labels) == pytest.approx(1 / 4)


def test_resolve_pos_weight_raises_on_unknown_mode():
    with pytest.raises(ClassBalancingError):
        resolve_pos_weight("bogus", None, [1, 0])
