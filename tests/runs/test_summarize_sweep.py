import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "summarize_sweep", Path(__file__).resolve().parents[2] / "scripts/vast/summarize_sweep.py"
)
summarize_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize_sweep)


def _snaps():
    rows = []
    for fold in (0, 1):
        for epoch, vl, vm, vf, fl, fm, ff in [(2, 0.40, 0.30, 0.70, 0.35, 0.40, 0.75),
                                              (4, 0.50, 0.35, 0.72, 0.20, 0.70, 0.85),
                                              (6, 0.90, 0.32, 0.74, 0.05, 0.95, 0.97)]:
            rows.append(dict(val_fold_id=fold, epoch=epoch, eval_group="val", loss=vl, mcc=vm, f1=vf, auroc=0.7))
            rows.append(dict(val_fold_id=fold, epoch=epoch, eval_group="fit", loss=fl, mcc=fm, f1=ff, auroc=0.9))
    return pd.DataFrame(rows)


def test_summarize_snapshots_decides_on_loss_mcc_f1():
    s = summarize_sweep.summarize_snapshots(_snaps())
    assert (s["best_loss"], s["@ep_loss"]) == (0.40, 2)      # lowest val loss
    assert (s["best_mcc"], s["@ep_mcc"]) == (0.35, 4)        # highest val MCC
    assert (s["best_f1"], s["@ep_f1"]) == (0.74, 6)          # highest val F1
    assert s["last_ep"] == 6 and s["last_loss"] == 0.90
    assert s["gap_loss"] == pytest.approx(0.85) and s["gap_mcc"] == pytest.approx(-0.63)  # val minus fit


def test_summarize_snapshots_reports_matched_metrics_when_present():
    snaps = _snaps()
    snaps["f1_matched"], snaps["pred_active_rate"] = 0.8, 0.6
    s = summarize_sweep.summarize_snapshots(snaps)
    assert s["last_f1_matched"] == 0.8 and s["last_pred_active_rate"] == 0.6
    assert "last_f1_matched" not in summarize_sweep.summarize_snapshots(_snaps())
