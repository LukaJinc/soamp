"""Isolated XGBoost worker: fit one model and score named matrices, in its OWN process.

Run as ``python src/soamp/engine/xgb_worker.py <job_dir>`` by ``soamp.engine.tree_cv``; it is a
plain script on purpose (no ``soamp`` imports, only numpy + xgboost) so the process never loads
torch or scikit-learn.

Why a separate process: XGBoost and PyTorch / scikit-learn each bring their own OpenMP runtime. On
macOS two runtimes in one process crash XGBoost (segmentation fault with torch loaded first, a hang
with all three) -- reproduced with xgboost 3.4.1 + Homebrew libomp. A worker sidesteps that on every
platform, and the parent process needs neither xgboost nor libomp.

Job directory in: ``job.json`` ({"params": XGBClassifier kwargs, "seed": int}), ``train.npz``
(x, y, optional w = sample weights) and ``sets.npz`` (named matrices to score).
Out: ``out.npz`` with, for every set ``s``: ``staged_s`` (rounds x rows, raw log-odds after each
boosting round) and ``final_s`` (the full model's log-odds), plus ``n_rounds``.

Per-round margins are built incrementally (each round's trees added to the previous margin through
``base_margin``), so scoring costs O(rounds), not O(rounds^2); the last round is checked against a
full prediction and the worker fails loudly if they ever disagree.
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(job_dir: str) -> None:
    import xgboost

    d = Path(job_dir)
    job = json.loads((d / "job.json").read_text())
    train, sets = np.load(d / "train.npz"), np.load(d / "sets.npz")
    model = xgboost.XGBClassifier(
        **job["params"], objective="binary:logistic", tree_method="hist", device="cpu",
        random_state=job["seed"], early_stopping_rounds=None, verbosity=0,
    )
    model.fit(train["x"], train["y"], sample_weight=train["w"] if "w" in train.files else None)
    booster = model.get_booster()
    n_rounds = int(booster.num_boosted_rounds())
    raw = json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]
    p0 = float(str(raw).strip("[]"))                       # xgboost 3.x writes it as "[7.03E-1]"
    base_margin = float(np.log(p0 / (1.0 - p0)))

    out = {"n_rounds": np.array(n_rounds)}
    for name in sets.files:
        x = sets[name]
        margin = np.full(len(x), base_margin, dtype=np.float64)
        staged = np.empty((n_rounds, len(x)), dtype=np.float64)
        for k in range(1, n_rounds + 1):
            margin = booster.predict(xgboost.DMatrix(x, base_margin=margin), output_margin=True, iteration_range=(k - 1, k))
            staged[k - 1] = margin
        final = np.asarray(booster.predict(xgboost.DMatrix(x), output_margin=True), dtype=np.float64)
        if not np.allclose(staged[-1], final, atol=1e-4):
            raise SystemExit(f"incremental margins disagree with the full prediction on set {name!r}: "
                             f"max diff {float(np.abs(staged[-1] - final).max()):.3g}")
        out[f"staged_{name}"], out[f"final_{name}"] = staged, final
    np.savez(d / "out.npz", **out)


if __name__ == "__main__":
    main(sys.argv[1])
