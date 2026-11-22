"""Per-fold fit/val metrics of the best run+epoch of each peptide representation (read-only on wandb).

Best = highest mean val MCC over all epoch snapshots of the representation's runs (the dev decision rule),
excluding the no-class-weight run (plain BCE, loss not comparable). Writes reports/dev_best_runs_folds.csv.

usage: python scripts/slides/build_best_run_folds.py
"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()
REPO_ROOT = Path(__file__).resolve().parents[2]
BEST = {"PeptideCLM": ("dev_sweep_wd1e-3", 10), "RDKit": ("dev_repr_rdkit_vocab_small", 86),
        "Molecular graph": ("dev_repr_molgraph_vocab_small", 64)}


def main() -> None:
    api, work, frames = wandb.Api(), tempfile.mkdtemp(), []
    for rep, (name, epoch) in BEST.items():
        run = next(r for r in api.runs("soamp", filters={"display_name": name}) if r.state == "finished")
        art = next(a for a in run.logged_artifacts() if a.name.startswith("cv_results"))
        art.download(root=os.path.join(work, run.id))
        tb = art.get("cv_snapshots")
        df = pd.DataFrame(tb.data, columns=tb.columns)
        df["epoch"] = df["epoch"].astype(int)
        df = df[df.epoch == epoch].copy()
        df.insert(0, "representation", rep)
        df.insert(1, "run", name)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(REPO_ROOT / "reports" / "dev_best_runs_folds.csv", index=False)
    print(out[["representation", "epoch", "val_fold_id", "eval_group", "mcc", "precision", "recall", "f1", "loss"]].round(3).to_string())


if __name__ == "__main__":
    main()
