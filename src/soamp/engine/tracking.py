"""Thin, wandb-only experiment-run setup shared between pipeline/train.py
and notebooks -- mirrors soamp.data.factory.build_dataset /
soamp.model.factory.build_model's ergonomics (one function call, real
values sourced from objects already in hand, not hand-typed twice).

Deliberately NOT a Protocol/multi-backend dispatch system like the removed
soamp.common.tracking -- wandb is the only backend in use; reintroduce that
abstraction only if a second backend actually becomes necessary.
"""
from typing import TYPE_CHECKING, Any

import wandb

if TYPE_CHECKING:
    from soamp.data.factory import DatasetBundle


def build_tracker(
    bundle: "DatasetBundle",
    hyperparams: dict[str, Any],
    *,
    project: str = "soamp",
    job_type: str | None = None,
    run_name: str | None = None,
    **wandb_init_kwargs: Any,
):
    """wandb.init() with config = hyperparams + the bundle's actual
    Featurization (peptide/organism representation method + dims) -- every
    caller logs real representation info without re-deriving or hand-typing
    it. `hyperparams` carries whatever's specific to the caller's run
    (epochs, hidden_dims, seed, batch_size, learning_rate, fold-generation
    metadata, etc.); featurization fields are authoritative and applied
    after the hyperparams spread, so they can never be silently overridden
    by a stale hand-typed value. `**wandb_init_kwargs` (e.g. `mode="disabled"`)
    passes straight through to `wandb.init` for testing.
    """
    featurization = bundle.featurization
    config = {
        **hyperparams,
        "peptide_method": featurization.peptide_method,
        "peptide_feature_dim": featurization.peptide_feature_dim,
        "descriptor_names": featurization.descriptor_names,
        "organism_method": featurization.organism_method,
        "organism_output_kind": featurization.organism_output_kind,
        "organism_vocab_size": featurization.organism_vocab_size,
        "organism_feature_dim": featurization.organism_feature_dim,
    }
    return wandb.init(
        project=project, job_type=job_type, name=run_name, config=config, **wandb_init_kwargs,
    )


def watch_model(model, *, log: str = "all", log_graph: bool = True, log_freq: int = 100) -> None:
    """Call once on the real model being trained, before its training loop
    starts, so gradient/parameter-weight histograms and the computation
    graph populate from real forward/backward passes -- never a throwaway
    preview model. `log="all"` = gradients + parameter weights (wandb has
    no built-in activation-value logging; not attempted here).
    """
    wandb.watch(model, log=log, log_graph=log_graph, log_freq=log_freq)
