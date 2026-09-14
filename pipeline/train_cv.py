"""
Cross-validation step: 5-fold CV over data/train_folds_leiden.csv's
Leiden-community folds, reproducing scripts/EDA/build_dataset_build_model.ipynb's
exploratory CV loop as a real, tested pipeline stage -- see
src/soamp/engine/cross_validation.py for the per-fold train+eval logic this
orchestrates, and its module docstring for why the fold file is joined on
`sequence` and not `node_id`/`peptide_id`.

This is a *different stage* from pipeline/train.py, not a second
implementation of the same training loop (CLAUDE.md sec 1): pipeline/train.py
produces the real held-out-test benchmark from one fixed split; this produces
a cross-fold generalization estimate over the train split only -- 'test' is
never touched here. Both share the same underlying primitives
(build_dataset, build_model, Trainer, compute_binary_metrics).

Config: pass --config to select a config/train/cv_*.yaml overlay -- each
`_base_`s off the matching config/train/exp_*.yaml featurization-grid cell,
so the same peptide/organism method and architecture are covered by both the
single-split benchmark and this CV estimate.

Output: reports/cv_results_<exp_id>.csv (row-level -- see
soamp.engine.cross_validation.train_and_evaluate_fold's docstring for exact
columns), reports/train_cv_<exp_id>_log.txt (summary), plus a wandb run
(job_type="kfold_cv") with per-fold history, an organism_activity_thresholds
artifact, cross-fold mean/std summary, and a cv_results Artifact.
"""
import argparse
import csv
import json

import pandas as pd
import wandb
from dotenv import load_dotenv

from soamp.data.factory import build_dataset
from soamp.engine.config import REPO_ROOT, TrainConfig
from soamp.engine.cross_validation import (
    assign_fold_ids,
    load_fold_assignments,
    summarize_cv_metrics,
    train_and_evaluate_fold,
)
from soamp.engine.tracking import build_tracker
from soamp.utils.config import load_config
from soamp.utils.device import resolve_device
from soamp.utils.logging import configure_logging
from soamp.utils.reproducibility import git_sha, seed_everything

load_dotenv()

METRIC_NAMES = ["accuracy", "f1", "precision", "recall", "auroc"]

# Hand-transcribed from scripts/EDA/generate_leiden_train_folds.ipynb -- that
# notebook builds train_folds_leiden.csv and isn't re-run here, so keep this
# in sync manually if its params ever change. Static provenance about a fixed
# committed artifact, not a per-run tunable -- see CVConfig for the one part
# of this (which folds file to read) that is config-driven.
FOLD_GENERATION_METADATA = {
    "method": "sequence-identity graph (blosum45 alignment) + Leiden community "
              "detection; folds = greedy cumulative-count binning of communities "
              "into n_folds groups",
    "identity_threshold": 0.60,
    "substitution_matrix": "blosum45",
    "gap_open": 5,
    "gap_extension": 1,
    "leiden_n_iterations": -1,
    "leiden_seed": 42,
    "n_folds": 5,
    "source_notebook": "scripts/EDA/generate_leiden_train_folds.ipynb",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    CFG = load_config(args.config, TrainConfig)
    LOG_PATH = CFG.paths.reports_dir / f"train_cv_{CFG.exp_id}_log.txt"
    RESULTS_CSV_PATH = CFG.paths.reports_dir / f"cv_results_{CFG.exp_id}.csv"

    log = configure_logging("train_cv")
    log.info(f"Loaded config: {args.config} (exp_id={CFG.exp_id})")
    seed_everything(CFG.loop.seed)
    device = resolve_device(CFG.loop.device)
    log.info(f"device: {device} (config: {CFG.loop.device!r})")

    classification_csv = CFG.paths.data_dir / CFG.input_files.classification_dataset_filename
    folds_csv = CFG.paths.data_dir / CFG.cv.folds_csv_filename

    with open(classification_csv, newline="") as f:
        all_rows = list(csv.DictReader(f))
    train_rows = [r for r in all_rows if r["split"] == "train"]
    fold_by_sequence = load_fold_assignments(folds_csv)
    train_rows = assign_fold_ids(train_rows, fold_by_sequence)
    fold_ids = sorted(set(r["fold_id"] for r in train_rows))
    log.info(f"{len(train_rows)} train rows across {len(fold_ids)} folds: {fold_ids}")

    # row_groups mode computes featurization fresh per fold rather than
    # reading a precomputed artifact, so it needs an explicit method name --
    # read it off the artifact files' own self-describing "method" field
    # (peeking at just that key, not the full feature/vocab payload) so the
    # method a CV run uses always matches the featurization grid cell its
    # exp_*.yaml base was built for, with no new config field to keep in sync.
    with open(CFG.paths.data_dir / CFG.input_files.peptide_feature_scaler_filename) as f:
        peptide_method = json.load(f)["method"]
    with open(CFG.paths.data_dir / CFG.input_files.organism_vocab_filename) as f:
        organism_method = json.load(f)["method"]
    log.info(f"peptide_method={peptide_method!r} organism_method={organism_method!r}")

    # PeptideCLM embeddings are frozen/deterministic and don't depend on
    # fold membership -- one dict shared across every build_dataset call
    # below (the preview call and all 5 folds', each constructing a fresh
    # PeptideCLMFeaturizer instance) means a peptide is embedded once,
    # however many folds/calls it appears in, rather than up to 6 times.
    # See PeptideCLMFeaturizer's docstring. A no-op for rdkit_descriptors.
    peptide_feature_cache: dict = {}
    peptide_method_kwargs = (
        {"cache": peptide_feature_cache} if peptide_method == "peptideclm_embedding" else None
    )

    # Built purely to source the real peptide/organism representation info
    # (method names, dims) build_tracker logs into the run config -- not
    # used for training/eval, since each fold below fits its own scaler/vocab
    # on that fold's own fit partition. Processes every unique peptide in
    # train_rows, which -- with the shared cache above -- means this call
    # alone does the one real PeptideCLM forward pass; every fold's call
    # below is then a cache hit.
    preview_bundle = build_dataset(
        row_groups={"fit": train_rows},
        peptide_method=peptide_method,
        peptide_method_kwargs=peptide_method_kwargs,
        organism_method=organism_method,
    )
    run = build_tracker(
        preview_bundle,
        hyperparams={
            "epochs": CFG.loop.epochs,
            "batch_size": CFG.loop.batch_size,
            "learning_rate": CFG.optim.learning_rate,
            "seed": CFG.loop.seed,
            "architecture": CFG.model.architecture,
            **CFG.model.active_kwargs(),
            "class_balancing_mode": CFG.class_balancing.mode,
            "device": str(device),
            "fold_generation": FOLD_GENERATION_METADATA,
        },
        project="soamp", job_type="kfold_cv", run_name=CFG.exp_id,
        group=CFG.wandb_group, tags=CFG.wandb_tags,
    )

    thresholds_df = pd.read_csv(REPO_ROOT / "config/thresholds/organism_thresholds.csv")
    organisms_used = sorted({r["organism"] for r in train_rows})
    organism_thresholds = thresholds_df[
        thresholds_df["match_key"].isin(organisms_used) & (thresholds_df["level"] == "species")
    ][["match_key", "active_threshold_uM", "inactive_threshold_uM", "source"]]
    thresholds_artifact = wandb.Artifact(name="organism_activity_thresholds", type="dataset")
    thresholds_artifact.add(wandb.Table(dataframe=organism_thresholds), "organism_thresholds")
    run.log_artifact(thresholds_artifact)

    fold_results = []
    fold_metrics_rows = []
    for fold_id in fold_ids:
        row_groups = {
            "fit": [r for r in train_rows if r["fold_id"] != fold_id],
            "val": [r for r in train_rows if r["fold_id"] == fold_id],
        }
        metrics_by_group, results_df = train_and_evaluate_fold(
            row_groups,
            eval_groups=("fit", "val"),
            epochs=CFG.loop.epochs,
            seed=CFG.loop.seed,
            peptide_method=peptide_method,
            peptide_method_kwargs=peptide_method_kwargs,
            organism_method=organism_method,
            architecture=CFG.model.architecture,
            architecture_kwargs=CFG.model.active_kwargs(),
            batch_size=CFG.loop.batch_size,
            learning_rate=CFG.optim.learning_rate,
            class_balancing_mode=CFG.class_balancing.mode,
            class_balancing_fixed_pos_weight=CFG.class_balancing.fixed_pos_weight,
            device=device,
            watch=True,
        )
        fit_m, val_m = metrics_by_group["fit"], metrics_by_group["val"]
        log.info(
            f"fold {fold_id}: fit_auroc={fit_m['auroc']:.4f} val_auroc={val_m['auroc']:.4f} "
            f"(n_fit={len(row_groups['fit'])} n_val={len(row_groups['val'])})"
        )
        run.log({
            "fold": fold_id,
            **{f"fit/{k}": v for k, v in fit_m.items()},
            **{f"val/{k}": v for k, v in val_m.items()},
            "n_fit": len(row_groups["fit"]), "n_val": len(row_groups["val"]),
        }, step=int(fold_id))

        results_df["val_fold_id"] = fold_id
        fold_results.append(results_df)
        for eval_group, m in metrics_by_group.items():
            fold_metrics_rows.append({"val_fold_id": fold_id, "eval_group": eval_group, **m})

    cv_results_df = pd.concat(fold_results, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metrics_rows)

    summary_df = summarize_cv_metrics(fold_metrics_df, METRIC_NAMES)
    log.info(f"mean +/- std across folds:\n{summary_df}")
    for eval_group in summary_df.index:
        for metric in METRIC_NAMES:
            run.summary[f"{eval_group}_{metric}_mean"] = summary_df.loc[eval_group, (metric, "mean")]
            run.summary[f"{eval_group}_{metric}_std"] = summary_df.loc[eval_group, (metric, "std")]

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    cv_results_df.to_csv(RESULTS_CSV_PATH, index=False)

    results_artifact = wandb.Artifact(name="cv_results", type="results")
    results_artifact.add(wandb.Table(dataframe=cv_results_df), "cv_results")
    logged_results_artifact = run.log_artifact(results_artifact)
    logged_results_artifact.wait()  # block until server-side commit, so the round-trip
                                     # fetch right below doesn't race the async upload

    api = wandb.Api()
    reloaded_artifact = api.artifact(f"{run.entity}/{run.project}/cv_results:latest")
    reloaded_table = reloaded_artifact.get("cv_results")
    reloaded_df = pd.DataFrame(reloaded_table.data, columns=reloaded_table.columns)
    assert reloaded_df.shape == cv_results_df.shape, (reloaded_df.shape, cv_results_df.shape)
    log.info(f"round-trip OK: reloaded {reloaded_df.shape} matches cv_results_df {cv_results_df.shape}")

    sha = git_sha()
    run.finish()

    log_lines = [
        f"=== Train CV: {CFG.exp_id} ===",
        f"git_sha: {sha}",
        f"seed: {CFG.loop.seed}",
        f"device: {device}",
        f"epochs per fold: {CFG.loop.epochs}",
        f"folds: {fold_ids}",
        f"cv_results_df: {RESULTS_CSV_PATH} ({len(cv_results_df)} rows)",
        f"mean +/- std across folds:\n{summary_df}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))


if __name__ == "__main__":
    main()
