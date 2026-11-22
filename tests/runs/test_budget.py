import pytest

from soamp.runs.budget import (
    CostModel,
    Limits,
    estimate,
    load_cost_model,
    load_limits,
    violations,
)
from soamp.runs.plan import Cell


def _cell(pep="rdkit", epochs=100, i=0) -> Cell:
    return Cell(f"c{i}", "baseline_classifier", pep, "vocab", 42, epochs, {})


LIM = Limits(max_cost_per_request_usd=5, max_concurrent_instances=2, max_cost_per_24h_usd=20,
             max_cells=40, max_hours_per_instance=3, safety_margin=1.25)
MODEL = CostModel(minutes_per_100_epochs={"rdkit": 6.0, "peptideclm": 60.0}, instance_overhead_minutes=0,
                  assumed_price_per_hour_usd=1.0)


def test_committed_files_load():
    assert load_limits().max_concurrent_instances >= 1
    assert "molgraph" in load_cost_model().minutes_per_100_epochs


def test_estimate_scales_with_epochs_and_margin():
    est = estimate([_cell(epochs=100)] * 10, MODEL, LIM)  # 10 x 6 min = 1 h
    assert est.cell_hours == pytest.approx(1.0)
    assert est.cost_usd == pytest.approx(1.25)
    assert estimate([_cell(epochs=50)] * 10, MODEL, LIM).cell_hours == pytest.approx(0.5)


def test_estimate_shards_only_as_far_as_needed_and_allowed():
    four_h = [_cell("peptideclm", 100, i) for i in range(4)]  # 4 h of work
    assert estimate(four_h, MODEL, LIM).n_instances == 2  # 2 h each <= 3 h cap
    six_h = [_cell("peptideclm", 100, i) for i in range(6)]
    est = estimate(six_h, MODEL, LIM)
    assert est.n_instances == 2 and est.hours_per_instance == pytest.approx(3.0)
    assert estimate(six_h, MODEL, LIM, instances_available=1).n_instances == 1


def test_allowed_request_has_no_violations():
    est = estimate([_cell()] * 5, MODEL, LIM)
    assert violations(est, LIM, instances_running=0, spend_last_24h_usd=0) == []


def test_each_cap_refuses():
    est = estimate([_cell()] * 5, MODEL, LIM)
    assert any("max_concurrent" in v for v in violations(est, LIM, instances_running=2, spend_last_24h_usd=0))
    assert any("24h" in v for v in violations(est, LIM, instances_running=0, spend_last_24h_usd=19.9))
    big = estimate([_cell("peptideclm", 100, i) for i in range(8)], MODEL, LIM)  # 8 h > 2 x 3 h
    msgs = violations(big, LIM, instances_running=0, spend_last_24h_usd=0)
    assert any("max_hours_per_instance" in v for v in msgs)
    pricey = estimate([_cell()] * 5, MODEL, LIM, price_per_hour_usd=100)
    assert any("max_cost_per_request" in v for v in violations(pricey, LIM, instances_running=0, spend_last_24h_usd=0))
    many = estimate([_cell(i=i) for i in range(41)], MODEL, LIM)
    assert any("max_cells" in v for v in violations(many, LIM, instances_running=0, spend_last_24h_usd=0))
    assert any("0 cells" in v for v in violations(estimate([], MODEL, LIM), LIM, instances_running=0, spend_last_24h_usd=0))


def test_limits_can_be_lowered_never_raised():
    assert LIM.lowered(max_cost_per_request_usd=1).max_cost_per_request_usd == 1
    assert LIM.lowered(safety_margin=2.0).safety_margin == 2.0
    with pytest.raises(ValueError):
        LIM.lowered(max_cost_per_request_usd=50)
    with pytest.raises(ValueError):
        LIM.lowered(safety_margin=1.0)
    with pytest.raises(KeyError):
        LIM.lowered(nonsense=1)


def test_missing_peptide_timing_raises():
    with pytest.raises(KeyError):
        estimate([_cell("molgraph")], MODEL, LIM)
