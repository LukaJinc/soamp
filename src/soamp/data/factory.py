"""Notebook-friendly composition of the dataset-assembly + featurization
building blocks into ready-to-train PeptideOrganismDataset objects.

Two modes, both routed through the same featurizer-registry/
PeptideOrganismDataset primitives:
  - row_groups=None (default): loads mic_classification_dataset.csv + the
    precomputed feature artifacts (peptide_features.csv, organism_vocab.json,
    peptide_feature_scaler.json, val_split.json) from data_dir, reproducing
    exactly what pipeline/train.py runs -- {"fit", "val", "test"} datasets.
  - row_groups={"fit": [...], ...}: an arbitrary in-memory partition (e.g.
    one k-fold iteration, or "fit on population A, eval on population B").
    Featurization is computed fresh via the selected peptide_method/
    organism_method, with the scaler and organism featurizer fit only on
    row_groups[fit_group] -- never on other groups' rows, to avoid leaking a
    fold's held-out peptides/organisms into its own scaler or OOV handling.

peptide_method/organism_method select a concrete strategy via
soamp.features.{peptide_featurizers,organism_featurizers}.build_*() --
picking a method here (or from config, in pipeline/train.py's case) is the
one place a caller controls "which featurization" without touching library
code, per CLAUDE.md's pluggable-featurization ask.

Both a notebook and pipeline/train.py call build_dataset directly, so
there is exactly one implementation of "how a Dataset gets built" per
CLAUDE.md sec 1.
"""
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from soamp.data.splitting import apply_val_split
from soamp.data.torch_dataset import PeptideOrganismDataset
from soamp.features.organism_featurizers import (
    OrganismFeaturizer,
    build_organism_featurizer,
    organism_featurizer_from_artifact,
)
from soamp.features.peptide import index_feature_rows_by_peptide_id, select_unique_peptides
from soamp.features.peptide_featurizers import PeptideFeaturizer, build_peptide_featurizer
from soamp.features.scaling import fit_scaler

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PEPTIDE_METHOD = "rdkit_descriptors"
DEFAULT_ORGANISM_METHOD = "vocab_embedding"


class DatasetFactoryError(ValueError):
    """Raised when row_groups is empty or doesn't contain fit_group, or
    when peptide_method/organism_method (or their kwargs) are overridden in
    artifact mode (where the on-disk scaler/vocab artifacts are
    authoritative)."""


@dataclass
class Featurization:
    descriptor_names: list[str]
    peptide_feature_dim: int
    scaler: dict  # {"descriptor_names", "mean", "scale"}
    organism_vocab: dict[str, int] | None
    unknown_index: int
    organism_vocab_size: int | None
    organism_output_kind: Literal["index", "vector"] = "index"
    organism_feature_dim: int | None = None
    peptide_method: str = DEFAULT_PEPTIDE_METHOD
    organism_method: str = DEFAULT_ORGANISM_METHOD


@dataclass
class DatasetBundle:
    datasets: dict[str, PeptideOrganismDataset]  # keyed by caller's group names
    featurization: Featurization


def _dataset_from_featurization(
    rows: list[dict],
    peptide_features: dict[str, dict[str, float]],
    featurization: Featurization,
    organism_featurizer: OrganismFeaturizer,
) -> PeptideOrganismDataset:
    return PeptideOrganismDataset(
        rows=rows,
        peptide_features=peptide_features,
        descriptor_names=featurization.descriptor_names,
        scaler_mean=featurization.scaler["mean"],
        scaler_scale=featurization.scaler["scale"],
        organism_featurizer=organism_featurizer,
    )


def build_dataset(
    row_groups: dict[str, list[dict]] | None = None,
    fit_group: str = "fit",
    *,
    peptide_method: str = DEFAULT_PEPTIDE_METHOD,
    peptide_method_kwargs: dict | None = None,
    organism_method: str = DEFAULT_ORGANISM_METHOD,
    organism_method_kwargs: dict | None = None,
    data_dir: Path | str | None = None,
    classification_dataset_filename: str = "mic_classification_dataset.csv",
    peptide_features_filename: str = "peptide_features.csv",
    organism_vocab_filename: str = "organism_vocab.json",
    peptide_feature_scaler_filename: str = "peptide_feature_scaler.json",
    val_split_filename: str = "val_split.json",
) -> DatasetBundle:
    """Builds a DatasetBundle either from precomputed pipeline artifacts
    (row_groups=None, the default -- exact parity with pipeline/train.py's
    fit/val/test split) or from an arbitrary in-memory row partition
    (row_groups={"fit": [...], "val": [...], ...}, any caller-chosen keys)
    with featurization computed fresh via peptide_method/organism_method,
    fit only on row_groups[fit_group].
    """
    if row_groups is None:
        if (
            peptide_method != DEFAULT_PEPTIDE_METHOD
            or peptide_method_kwargs is not None
            or organism_method != DEFAULT_ORGANISM_METHOD
            or organism_method_kwargs is not None
        ):
            raise DatasetFactoryError(
                "peptide_method/organism_method (and their kwargs) are not "
                "accepted in artifact mode (row_groups=None) -- the on-disk "
                "scaler/vocab artifacts are authoritative there"
            )
        return _build_dataset_from_artifacts(
            data_dir=data_dir,
            classification_dataset_filename=classification_dataset_filename,
            peptide_features_filename=peptide_features_filename,
            organism_vocab_filename=organism_vocab_filename,
            peptide_feature_scaler_filename=peptide_feature_scaler_filename,
            val_split_filename=val_split_filename,
        )
    return _build_dataset_from_rows(
        row_groups,
        fit_group,
        peptide_method=peptide_method,
        peptide_method_kwargs=peptide_method_kwargs or {},
        organism_method=organism_method,
        organism_method_kwargs=organism_method_kwargs or {},
    )


def _build_dataset_from_artifacts(
    *,
    data_dir: Path | str | None,
    classification_dataset_filename: str,
    peptide_features_filename: str,
    organism_vocab_filename: str,
    peptide_feature_scaler_filename: str,
    val_split_filename: str,
) -> DatasetBundle:
    resolved_data_dir = Path(data_dir) if data_dir is not None else REPO_ROOT / "data"

    with open(resolved_data_dir / classification_dataset_filename, newline="") as f:
        all_rows = list(csv.DictReader(f))
    with open(resolved_data_dir / peptide_features_filename, newline="") as f:
        peptide_feature_rows = list(csv.DictReader(f))
    with open(resolved_data_dir / organism_vocab_filename) as f:
        vocab_data = json.load(f)
    with open(resolved_data_dir / peptide_feature_scaler_filename) as f:
        scaler = json.load(f)
    with open(resolved_data_dir / val_split_filename) as f:
        val_split = json.load(f)

    descriptor_names = scaler["descriptor_names"]
    peptide_features = index_feature_rows_by_peptide_id(peptide_feature_rows, descriptor_names)

    # organism_vocab.json is self-describing via its "method" field (same
    # principle as the peptide side's scaler artifact) -- no filename
    # bifurcation needed to support a second organism method.
    organism_featurizer = organism_featurizer_from_artifact(vocab_data)

    featurization = Featurization(
        descriptor_names=descriptor_names,
        peptide_feature_dim=len(descriptor_names),
        scaler=scaler,
        organism_vocab=getattr(organism_featurizer, "vocab", None),
        unknown_index=getattr(organism_featurizer, "unknown_index", 0),
        organism_vocab_size=organism_featurizer.vocab_size,
        organism_output_kind=organism_featurizer.output_kind,
        organism_feature_dim=organism_featurizer.feature_dim,
        peptide_method=scaler.get("method", DEFAULT_PEPTIDE_METHOD),
        organism_method=vocab_data.get("method", DEFAULT_ORGANISM_METHOD),
    )

    train_rows = [r for r in all_rows if r["split"] == "train"]
    test_rows = [r for r in all_rows if r["split"] == "test"]
    val_peptide_ids = set(val_split["val_peptide_ids"])
    fit_rows, val_rows = apply_val_split(train_rows, val_peptide_ids)

    datasets = {
        "fit": _dataset_from_featurization(fit_rows, peptide_features, featurization, organism_featurizer),
        "val": _dataset_from_featurization(val_rows, peptide_features, featurization, organism_featurizer),
        "test": _dataset_from_featurization(test_rows, peptide_features, featurization, organism_featurizer),
    }
    return DatasetBundle(datasets=datasets, featurization=featurization)


def _build_dataset_from_rows(
    row_groups: dict[str, list[dict]],
    fit_group: str,
    *,
    peptide_method: str,
    peptide_method_kwargs: dict,
    organism_method: str,
    organism_method_kwargs: dict,
) -> DatasetBundle:
    if not row_groups:
        raise DatasetFactoryError("row_groups is empty")
    if fit_group not in row_groups:
        raise DatasetFactoryError(
            f"fit_group={fit_group!r} is not a key of row_groups ({list(row_groups)})"
        )

    peptide_featurizer: PeptideFeaturizer = build_peptide_featurizer(
        peptide_method, **peptide_method_kwargs
    )

    all_rows = [row for rows in row_groups.values() for row in rows]
    unique_peptides = select_unique_peptides(all_rows)

    fit_rows = row_groups[fit_group]
    fit_peptide_ids = {r["peptide_id"] for r in fit_rows}
    fit_unique_peptides = [p for p in unique_peptides if p["peptide_id"] in fit_peptide_ids]

    peptide_featurizer.fit(fit_unique_peptides)
    feature_rows = peptide_featurizer.transform(unique_peptides)
    descriptor_names = peptide_featurizer.feature_names
    peptide_features = index_feature_rows_by_peptide_id(feature_rows, descriptor_names)

    fit_feature_rows = [row for row in feature_rows if row["peptide_id"] in fit_peptide_ids]
    scaler = fit_scaler(fit_feature_rows, descriptor_names)

    organism_featurizer: OrganismFeaturizer = build_organism_featurizer(
        organism_method, **organism_method_kwargs
    )
    fit_organisms = [r["organism"] for r in fit_rows]
    organism_featurizer.fit(fit_organisms)

    featurization = Featurization(
        descriptor_names=descriptor_names,
        peptide_feature_dim=len(descriptor_names),
        scaler=scaler,
        organism_vocab=getattr(organism_featurizer, "vocab", None),
        unknown_index=getattr(organism_featurizer, "unknown_index", 0),
        organism_vocab_size=organism_featurizer.vocab_size,
        organism_output_kind=organism_featurizer.output_kind,
        organism_feature_dim=organism_featurizer.feature_dim,
        peptide_method=peptide_method,
        organism_method=organism_method,
    )

    datasets = {
        name: _dataset_from_featurization(rows, peptide_features, featurization, organism_featurizer)
        for name, rows in row_groups.items()
    }
    return DatasetBundle(datasets=datasets, featurization=featurization)
