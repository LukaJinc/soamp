import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "build_prod_vs_dev_facts", Path(__file__).resolve().parents[2] / "scripts/slides/build_prod_vs_dev_facts.py"
)
facts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(facts)


def test_val_metrics_at_epochs_means_over_folds_and_matched_rate():
    rng = np.random.default_rng(5)
    frames = []
    for epoch, shift in ((5, -2.0), (100, 0.0)):         # same ranking, calibration shifted at epoch 5
        for fold in (0, 1):
            y = (rng.random(300) < 0.8).astype(int)
            x = y * 1.5 + rng.normal(size=300)
            frames.append(pd.DataFrame({"val_fold_id": fold, "epoch": epoch, "true_label_int": y, "logit": x + shift}))
    res = facts.val_metrics_at_epochs(pd.concat(frames, ignore_index=True)).set_index("epoch")
    assert set(res.index) == {5, 100}
    assert res.loc[5, "pred_active_rate"] < res.loc[100, "pred_active_rate"]          # shifted down at 0.5
    assert res.loc[5, "true_active_rate"] == pytest.approx(0.8, abs=0.06)
