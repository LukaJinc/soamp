"""
Train step: builds the peptide/organism Dataset and the model architecture
named by the config, and runs the full train/val/test loop, checkpointing
per CLAUDE.md sec 4 and writing overall + per-organism test metrics.

Consumes data/mic_classification_dataset.csv (dataset:model_ready) plus the
three feature artifacts named in the config's `input_files` block -- the
peptide feature CSV, its scaler, and the organism artifact (all from
pipeline/features/01-03, which must be run first with a matching
config/features/*.yaml). Those filenames are method-suffixed, so which
featurization a run uses is a config choice, not a source edit; the
artifacts are self-describing, so the tracker logs the real methods rather
than anything hand-typed here.

'test' is touched exactly once, at the very end, for the reported
metrics -- per-epoch monitoring and best-checkpoint selection use a
validation slice of 'train' read from data/val_split.json (produced by
pipeline/data/02_split_train_validation.py, the single source of truth
for this split -- see that script for how it's derived), never 'test'
itself.

Config: pass --config to select a config/train/*.yaml other than the
default base.yaml -- e.g. one of the exp_*.yaml featurization-grid cells.

Device comes from loop.device ("auto" -> CUDA when available). The model
reported on is the one selected by checkpoint.best_metric, reloaded from
<exp_id>_best.pth before the single test-split evaluation.

Output: reports/checkpoints/<exp_id>_*.pth,
reports/train_<exp_id>_log.txt (final metrics summary).
"""
import argparse

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from soamp.data.factory import build_dataset
from soamp.data.torch_dataset import LABEL_TO_INT
from soamp.engine.checkpointer import Checkpointer, CheckpointMetadata
from soamp.engine.class_balancing import resolve_pos_weight
from soamp.engine.config import TrainConfig
from soamp.engine.metrics import compute_binary_metrics, compute_metrics_by_organism
from soamp.engine.tracking import build_tracker, watch_model
from soamp.engine.trainer import Trainer
from soamp.model.factory import build_model
from soamp.utils.config import load_config
from soamp.utils.device import resolve_device
from soamp.utils.logging import configure_logging
from soamp.utils.reproducibility import git_sha, seed_everything

load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    CFG = load_config(args.config, TrainConfig)
    LOG_PATH = CFG.paths.reports_dir / f"train_{CFG.exp_id}_log.txt"

    log = configure_logging("train")
    log.info(f"Loaded config: {args.config} (exp_id={CFG.exp_id})")
    seed_everything(CFG.loop.seed)
    device = resolve_device(CFG.loop.device)
    log.info(f"device: {device} (config: {CFG.loop.device!r})")

    bundle = build_dataset(
        data_dir=CFG.paths.data_dir,
        classification_dataset_filename=CFG.input_files.classification_dataset_filename,
        peptide_features_filename=CFG.input_files.peptide_features_filename,
        organism_vocab_filename=CFG.input_files.organism_vocab_filename,
        peptide_feature_scaler_filename=CFG.input_files.peptide_feature_scaler_filename,
        val_split_filename=CFG.input_files.val_split_filename,
    )
    fit_ds, val_ds, test_ds = bundle.datasets["fit"], bundle.datasets["val"], bundle.datasets["test"]
    log.info(f"fit={len(fit_ds)} val={len(val_ds)} test={len(test_ds)} rows")

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

    model = build_model(
        bundle,
        architecture=CFG.model.architecture,
        **CFG.model.active_kwargs(),
    )

    run = build_tracker(
        bundle,
        hyperparams={
            "epochs": CFG.loop.epochs, "batch_size": CFG.loop.batch_size,
            "seed": CFG.loop.seed, "learning_rate": CFG.optim.learning_rate,
            "weight_decay": CFG.optim.weight_decay,
            "architecture": CFG.model.architecture,
            **CFG.model.active_kwargs(),
            "class_balancing_mode": CFG.class_balancing.mode,
            "device": str(device),
        },
        project="soamp", job_type="train", run_name=CFG.exp_id,
        group=CFG.wandb_group, tags=CFG.wandb_tags,
    )
    watch_model(model)

    fit_labels = [LABEL_TO_INT[r["label"]] for r in fit_ds.rows]
    pos_weight = resolve_pos_weight(
        CFG.class_balancing.mode, CFG.class_balancing.fixed_pos_weight, fit_labels
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device) if pos_weight is not None else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG.optim.learning_rate, weight_decay=CFG.optim.weight_decay
    )

    trainer = Trainer(model, optimizer, loss_fn, device=device)
    checkpointer = Checkpointer(CFG.paths.checkpoint_dir, CFG.exp_id)
    sha = git_sha()

    # Checkpoint selection is driven by CFG.checkpoint.best_metric (one of
    # val_auroc/val_accuracy/val_f1/val_loss), with direction taken from the
    # config's lower_is_better -- so switching the selection metric is a
    # config change, not a source edit.
    lower_is_better = CFG.checkpoint.lower_is_better
    best_val_metric = float("inf") if lower_is_better else float("-inf")
    best_epoch = None
    best_checkpoint_path = None
    for epoch in range(1, CFG.loop.epochs + 1):
        train_metrics = trainer.train_epoch(fit_loader)
        val_out = trainer.evaluate(val_loader)
        val_metrics = compute_binary_metrics(val_out["logits"], val_out["labels"])

        log.info(
            f"epoch {epoch}/{CFG.loop.epochs} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_out['loss']:.4f} val_auroc={val_metrics['auroc']:.4f}"
        )
        epoch_metrics = {
            "train_loss": train_metrics["loss"], "val_loss": val_out["loss"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        run.log(epoch_metrics, step=epoch)

        metadata = CheckpointMetadata(
            exp_id=CFG.exp_id, iter=epoch, git_sha=sha, seed=CFG.loop.seed,
            resolved_config=CFG.model_dump(mode="json"),
        )
        selection_value = epoch_metrics[CFG.checkpoint.best_metric]
        improved = (
            selection_value < best_val_metric if lower_is_better
            else selection_value > best_val_metric
        )
        if improved:
            best_val_metric = selection_value
            best_epoch = epoch
            best_checkpoint_path = checkpointer.save_weights(model, epoch, metadata, best=True)
        if epoch % CFG.checkpoint.save_every_n_epochs == 0:
            checkpointer.save_weights(model, epoch, metadata)
            checkpointer.save_full_state(model, optimizer, epoch, metadata)

    # Test metrics describe the *selected* model, not whichever weights the
    # last epoch happened to leave behind -- otherwise comparing two runs
    # partly measures which one overfit less by the final epoch.
    Checkpointer.load_weights(best_checkpoint_path, model)
    log.info(
        f"reloaded {best_checkpoint_path.name} for test eval "
        f"(best {CFG.checkpoint.best_metric}={best_val_metric:.4f} @ epoch {best_epoch})"
    )

    test_out = trainer.evaluate(test_loader)
    test_metrics = compute_binary_metrics(test_out["logits"], test_out["labels"])
    # Bucketed by the rows' own organism column rather than the model's encoded
    # organism input, so this works for both organism_output_kind values --
    # "vector" strategies (kmer_composition) have no vocab to invert. Valid
    # because test_loader is built with shuffle=False, so order is preserved.
    test_organism_names = [row["organism"] for row in test_ds.rows]
    test_metrics_by_org = compute_metrics_by_organism(
        test_out["logits"], test_out["labels"], test_organism_names
    )

    run.summary.update({f"test_{k}": v for k, v in test_metrics.items()})
    run.summary.update({
        f"test_{org}_{k}": v for org, m in test_metrics_by_org.items() for k, v in m.items()
    })
    run.finish()

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [
        f"=== Train: {CFG.exp_id} ===",
        f"git_sha: {sha}",
        f"seed: {CFG.loop.seed}",
        f"epochs run: {CFG.loop.epochs}",
        f"device: {device}",
        f"best {CFG.checkpoint.best_metric}: {best_val_metric:.4f} (epoch {best_epoch})",
        f"test eval checkpoint: {best_checkpoint_path.name}",
        f"checkpoint_dir: {CFG.paths.checkpoint_dir}",
        f"test rows: {len(test_ds)}",
        f"test metrics (overall): {test_metrics}",
        f"test metrics (per organism): {test_metrics_by_org}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))


if __name__ == "__main__":
    main()
