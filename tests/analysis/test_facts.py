import numpy as np
import pandas as pd
import pytest

from soamp.analysis.facts import RunData, analyze, facts_hash, load_thresholds, tidy

EPOCHS = [2, 4, 6, 8, 10]


def _run(name, *, val_mcc, fit_mcc=None, val_loss=None, val_f1=None, val_f1m=None, noise=0.0, seed=42, cb="auto", n_params=1000.0, fold_shift=0.0):
    """Synthetic run: each metric is a list over EPOCHS; folds differ by +-noise (deterministic)."""
    fit_mcc = fit_mcc or val_mcc
    val_loss = val_loss or [0.5] * len(EPOCHS)
    val_f1 = val_f1 or [0.7] * len(EPOCHS)
    val_f1m = val_f1m or val_f1
    rows = []
    for fold in range(5):
        off = (fold - 2) * noise + fold_shift
        for i, ep in enumerate(EPOCHS):
            rows.append({"val_fold_id": fold, "epoch": ep, "eval_group": "val", "mcc": val_mcc[i] + off, "loss": val_loss[i],
                         "f1": val_f1[i], "f1_matched": val_f1m[i], "precision": .7, "recall": .7, "auroc": .7, "accuracy": .7})
            rows.append({"val_fold_id": fold, "epoch": ep, "eval_group": "fit", "mcc": fit_mcc[i], "loss": 0.3,
                         "f1": .8, "f1_matched": .8, "precision": .8, "recall": .8, "auroc": .8, "accuracy": .8})
    return RunData(name, f"id-{name}", {"architecture": "baseline_classifier", "seed": seed, "class_balancing_mode": cb, "learning_rate": 1e-3},
                   n_params, pd.DataFrame(rows))


FLAT = [0.30] * 5
TH = load_thresholds()


def _kinds(facts, run):
    return {f["kind"] for f in facts["flags"] if f["run"] == run}


def test_tidy_has_mean_std_and_fold_count():
    t = tidy([_run("a", val_mcc=FLAT, noise=0.02)])
    row = t[(t.eval_group == "val") & (t.metric == "mcc") & (t.epoch == 10)].iloc[0]
    assert row["mean"] == pytest.approx(0.30) and row["n_folds"] == 5 and row["std"] > 0


def test_ranking_orders_by_consensus_of_loss_mcc_f1():
    good = _run("good", val_mcc=[.3, .35, .4, .4, .4], val_loss=[.5, .45, .4, .4, .4], val_f1=[.7, .75, .8, .8, .8])
    bad = _run("bad", val_mcc=FLAT, val_loss=[.6] * 5, val_f1=[.6] * 5)
    f, _ = analyze([bad, good], thresholds=TH)
    assert [r["run"] for r in f["ranking"]] == ["good", "bad"]
    assert f["ranking"][0]["rank_val_mcc"] == 1 and f["ranking"][0]["consensus"] == 1.0


def test_overfit_gap_flag_only_above_threshold():
    over = _run("over", val_mcc=FLAT, fit_mcc=[0.7] * 5)
    ok = _run("ok", val_mcc=FLAT, fit_mcc=[0.4] * 5)
    f, _ = analyze([over, ok], thresholds=TH)
    assert "overfit_gap" in _kinds(f, "over") and "overfit_gap" not in _kinds(f, "ok")


def test_val_loss_rises_flag():
    rising = _run("rise", val_mcc=FLAT, val_loss=[.50, .40, .45, .55, .65])
    flat = _run("flat", val_mcc=FLAT, val_loss=[.50, .49, .48, .47, .46])
    f, _ = analyze([rising, flat], thresholds=TH)
    assert "val_loss_rises" in _kinds(f, "rise") and "val_loss_rises" not in _kinds(f, "flat")


def test_calibration_drift_flag_needs_f1_up_with_matched_flat():
    drift = _run("drift", val_mcc=FLAT, val_f1=[.60, .65, .70, .74, .78], val_f1m=[.80, .80, .80, .79, .80])
    learn = _run("learn", val_mcc=FLAT, val_f1=[.60, .65, .70, .74, .78], val_f1m=[.60, .65, .70, .74, .78])
    f, _ = analyze([drift, learn], thresholds=TH)
    assert "calibration_drift" in _kinds(f, "drift") and "calibration_drift" not in _kinds(f, "learn")


def test_mixed_class_weighting_excludes_loss_from_ranking_and_warns():
    a = _run("a", val_mcc=FLAT, val_loss=[.3] * 5, cb="auto")
    b = _run("b", val_mcc=[.35] * 5, val_loss=[.9] * 5, cb="none")
    f, _ = analyze([a, b], thresholds=TH)
    assert {g["id"] for g in f["global_flags"]} >= {"mixed_class_weighting", "single_seed"}
    assert all(r["rank_val_loss"] is None for r in f["ranking"])
    assert f["ranking"][0]["run"] == "b"  # decided by MCC/F1 only


def test_paired_comparison_tie_vs_better_uses_fold_noise():
    base = _run("base", val_mcc=FLAT, noise=0.05)
    tie = _run("tie", val_mcc=[0.31] * 5, noise=0.05)         # +0.01 << fold std 0.07
    better = _run("better", val_mcc=[0.45] * 5, noise=0.05)    # +0.15 >> fold std
    f, _ = analyze([base, tie, better], baseline="base", thresholds=TH)
    v = {x["run"]: x["delta_mcc"] for x in f["vs_baseline"]}
    assert v["tie"]["verdict"] == "tie" and v["better"]["verdict"] == "better"
    assert v["better"]["folds_better"] == 5 and v["better"]["n_folds"] == 5


def test_missing_baseline_falls_back_to_best_run_and_failed_cells_become_flags():
    f, _ = analyze([_run("a", val_mcc=FLAT), _run("b", val_mcc=[.4] * 5)], baseline="not-there", thresholds=TH,
                   failed_cells=[{"cell": "c9", "state": "crashed", "detail": "rc=1"}])
    assert f["baseline"] == "b"
    assert any(g["id"] == "failed:c9" for g in f["global_flags"])


def test_facts_are_deterministic():
    runs = lambda: [_run("a", val_mcc=FLAT, noise=0.02), _run("b", val_mcc=[.4] * 5)]
    assert facts_hash(analyze(runs(), thresholds=TH)[0]) == facts_hash(analyze(runs(), thresholds=TH)[0])


def test_param_outlier_is_informational():
    f, _ = analyze([_run("a", val_mcc=FLAT, n_params=100), _run("b", val_mcc=FLAT, n_params=120), _run("big", val_mcc=FLAT, n_params=100000)], thresholds=TH)
    assert "param_outlier" in _kinds(f, "big") and "param_outlier" not in _kinds(f, "a")


def test_tied_values_share_a_rank_independent_of_input_order():
    a = _run("a", val_mcc=FLAT)
    b = _run("b", val_mcc=FLAT)
    for runs in ([a, b], [b, a]):
        f, _ = analyze(runs, thresholds=TH)
        assert {r["run"]: r["rank_val_mcc"] for r in f["ranking"]} == {"a": 1, "b": 1}
