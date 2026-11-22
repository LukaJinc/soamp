import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "build_dev_experiment_facts", Path(__file__).resolve().parents[2] / "scripts/slides/build_dev_experiment_facts.py"
)
facts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(facts)


def test_weighted_bce_matches_torch_batch_mean():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=600).astype(np.float32)
    labels = (rng.random(600) < 0.7).astype(np.float32)
    pw = 0.4
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
    expected = np.mean([
        loss_fn(torch.tensor(logits[i:i + 256]), torch.tensor(labels[i:i + 256])).item() for i in range(0, 600, 256)
    ])
    assert facts.weighted_bce_batch_mean(logits, labels, pw) == pytest.approx(expected, rel=1e-5)


def test_metrics_from_rows_and_aggregate():
    rng = np.random.default_rng(1)
    rows = []
    for fold in (0, 1):
        for group, n in (("fit", 80), ("val", 40)):
            y = (rng.random(n) < 0.6).astype(int)
            x = y * 1.5 - 0.5 + rng.normal(size=n)
            rows.append(pd.DataFrame({"val_fold_id": fold, "eval_group": group, "logit": x, "true_label_int": y}))
    snaps = facts.metrics_from_rows(pd.concat(rows, ignore_index=True), epoch=5)
    assert set(snaps.eval_group) == {"fit", "val"} and set(facts.METRICS) <= set(snaps.columns)
    agg = facts.aggregate(snaps, "run_x", "grp")
    row = agg[(agg.eval_group == "val") & (agg.metric == "mcc")].iloc[0]
    assert row.n_folds == 2 and row["run"] == "run_x"
    assert row["mean"] == pytest.approx(snaps[snaps.eval_group == "val"].mcc.mean())
