"""
Train step: builds the peptide/organism Dataset, the BaselineClassifier,
and runs the full train/val/test loop, checkpointing per CLAUDE.md sec 4
and writing overall + per-organism test metrics.

Consumes data/mic_classification_dataset.csv (dataset:model_ready) +
data/peptide_features.csv, data/organism_vocab.json,
data/peptide_feature_scaler.json (all from pipeline/features/01-03, must
be run first).

'test' is touched exactly once, at the very end, for the reported
metrics -- per-epoch monitoring and best-checkpoint selection use a
validation slice carved out of 'train' by peptide_id
(soamp.data.splitting.split_train_validation), never 'test' itself.

Config: pass --config to select a config/train/*.yaml other than the
default base.yaml -- e.g. --config config/train/wandb_run.yaml to log to
wandb instead of local files (see config/train/wandb_run.yaml, which sets
tracking.backend: wandb via _base_: base.yaml composition). The wandb
backend reads WANDB_API_KEY (and optionally WANDB_ENTITY) from the
process environment, loaded here from .env via python-dotenv -- see
.env.example. Never hardcode a real API key in a committed config file.

Output: reports/checkpoints/<exp_id>_*.pth,
reports/train_runs/<exp_id>/{metrics.jsonl,run_metadata.json},
reports/train_<exp_id>_log.txt (final metrics summary).
"""
import argparse
import csv
import json

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from soamp.common.tracking import build_tracker
from soamp.data.splitting import split_train_validation
from soamp.data.torch_dataset import LABEL_TO_INT, PeptideOrganismDataset
from soamp.engine.checkpointer import Checkpointer, CheckpointMetadata
from soamp.engine.class_balancing import resolve_pos_weight
from soamp.engine.config import TrainConfig
from soamp.engine.metrics import compute_binary_metrics, compute_metrics_by_organism
from soamp.engine.trainer import Trainer
from soamp.model.baseline_mlp import BaselineClassifier
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging
from soamp.utils.reproducibility import git_sha, seed_everything

load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train/base.yaml")
    return parser.parse_args()


def _build_dataset(rows, peptide_features, descriptor_names, scaler, vocab_data):
    return PeptideOrganismDataset(
        rows=rows,
        peptide_features=peptide_features,
        descriptor_names=descriptor_names,
        scaler_mean=scaler["mean"],
        scaler_scale=scaler["scale"],
        organism_vocab=vocab_data["vocab"],
        unknown_index=vocab_data["unknown_index"],
    )


def main() -> None:
    args = _parse_args()
    CFG = load_config(args.config, TrainConfig)
    CLASSIFICATION_CSV = CFG.paths.data_dir / CFG.input_files.classification_dataset_filename
    PEPTIDE_FEATURES_CSV = CFG.paths.data_dir / CFG.input_files.peptide_features_filename
    ORGANISM_VOCAB_JSON = CFG.paths.data_dir / CFG.input_files.organism_vocab_filename
    SCALER_JSON = CFG.paths.data_dir / CFG.input_files.peptide_feature_scaler_filename
    LOG_PATH = CFG.paths.reports_dir / f"train_{CFG.tracking.exp_id}_log.txt"

    log = configure_logging("train")
    log.info(f"Loaded config: {args.config} (tracking.backend={CFG.tracking.backend})")
    seed_everything(CFG.loop.seed)

    with open(CLASSIFICATION_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))
    with open(PEPTIDE_FEATURES_CSV, newline="") as f:
        peptide_feature_rows = list(csv.DictReader(f))
    with open(ORGANISM_VOCAB_JSON) as f:
        vocab_data = json.load(f)
    with open(SCALER_JSON) as f:
        scaler = json.load(f)

    descriptor_names = scaler["descriptor_names"]
    peptide_features = {
        row["peptide_id"]: {name: float(row[name]) for name in descriptor_names}
        for row in peptide_feature_rows
    }

    train_rows = [r for r in all_rows if r["split"] == "train"]
    test_rows = [r for r in all_rows if r["split"] == "test"]
    fit_rows, val_rows = split_train_validation(train_rows, CFG.split.val_fraction, CFG.loop.seed)
    log.info(f"fit={len(fit_rows)} val={len(val_rows)} test={len(test_rows)} rows")

    fit_ds = _build_dataset(fit_rows, peptide_features, descriptor_names, scaler, vocab_data)
    val_ds = _build_dataset(val_rows, peptide_features, descriptor_names, scaler, vocab_data)
    test_ds = _build_dataset(test_rows, peptide_features, descriptor_names, scaler, vocab_data)

    fit_loader = DataLoader(
        fit_ds, batch_size=CFG.loop.batch_size, shuffle=True,
        num_workers=CFG.loop.num_dataloader_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.loop.batch_size, num_workers=CFG.loop.num_dataloader_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG.loop.batch_size, num_workers=CFG.loop.num_dataloader_workers,
    )

    peptide_feature_dim = len(descriptor_names)
    organism_vocab_size = vocab_data["vocab_size"]
    model = BaselineClassifier(
        peptide_feature_dim=peptide_feature_dim,
        organism_vocab_size=organism_vocab_size,
        organism_embed_dim=CFG.model.organism_embed_dim,
        hidden_dims=CFG.model.hidden_dims,
    )

    fit_labels = [LABEL_TO_INT[r["label"]] for r in fit_rows]
    pos_weight = resolve_pos_weight(
        CFG.class_balancing.mode, CFG.class_balancing.fixed_pos_weight, fit_labels
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight) if pos_weight is not None else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG.optim.learning_rate, weight_decay=CFG.optim.weight_decay
    )

    trainer = Trainer(model, optimizer, loss_fn)
    checkpointer = Checkpointer(CFG.paths.checkpoint_dir, CFG.tracking.exp_id)
    tracker = build_tracker(CFG.tracking, CFG.paths.tracking_dir)
    tracker.log_config(CFG.model_dump(mode="json"))
    sha = git_sha()

    best_val_auroc = float("-inf")
    best_epoch = None
    best_checkpoint_path = None
    for epoch in range(1, CFG.loop.epochs + 1):
        train_metrics = trainer.train_epoch(fit_loader)
        val_out = trainer.evaluate(val_loader)
        val_metrics = compute_binary_metrics(val_out["logits"], val_out["labels"])

        tracker.log_metrics(
            {
                "train_loss": train_metrics["loss"],
                "val_loss": val_out["loss"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
            },
            step=epoch,
        )
        log.info(
            f"epoch {epoch}/{CFG.loop.epochs} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_out['loss']:.4f} val_auroc={val_metrics['auroc']:.4f}"
        )

        metadata = CheckpointMetadata(
            exp_id=CFG.tracking.exp_id, iter=epoch, git_sha=sha, seed=CFG.loop.seed,
            resolved_config=CFG.model_dump(mode="json"),
        )
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            best_checkpoint_path = checkpointer.save_weights(model, epoch, metadata, best=True)
        if epoch % CFG.checkpoint.save_every_n_epochs == 0:
            checkpointer.save_weights(model, epoch, metadata)
            checkpointer.save_full_state(model, optimizer, epoch, metadata)

    test_out = trainer.evaluate(test_loader)
    test_metrics = compute_binary_metrics(test_out["logits"], test_out["labels"])
    index_to_organism = {v: k for k, v in vocab_data["vocab"].items()}
    test_metrics_by_org = compute_metrics_by_organism(
        test_out["logits"], test_out["labels"], test_out["organism_idx"], index_to_organism
    )
    tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()}, step=CFG.loop.epochs)
    tracker.log_artifact(
        name=f"{CFG.tracking.exp_id}_model",
        artifact_type="model",
        paths=[best_checkpoint_path],
        metadata={
            "best_epoch": best_epoch, "best_val_auroc": best_val_auroc, "git_sha": sha,
            "seed": CFG.loop.seed, "test_metrics": test_metrics,
        },
        depends_on=["dataset_model_ready", "features_peptide", "features_organism", "features_peptide_scaler"],
    )
    tracker.close()

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [
        f"=== Train: {CFG.tracking.exp_id} ===",
        f"git_sha: {sha}",
        f"seed: {CFG.loop.seed}",
        f"epochs run: {CFG.loop.epochs}",
        f"best val_auroc: {best_val_auroc:.4f} (epoch {best_epoch})",
        f"checkpoint_dir: {CFG.paths.checkpoint_dir}",
        f"tracking_dir: {CFG.paths.tracking_dir / CFG.tracking.exp_id}",
        f"test rows: {len(test_rows)}",
        f"test metrics (overall): {test_metrics}",
        f"test metrics (per organism): {test_metrics_by_org}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))


if __name__ == "__main__":
    main()
