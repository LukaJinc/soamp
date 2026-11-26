"""Linear-only subset of an existing classification dataset (pure logic).

The hypothesis this serves: does a model trained on linear peptides alone do better than one
trained on the full mixture of linear, disulfide-bridged and cyclic peptides? The subset is a
row filter on a dataset that is already built, so the split assignment of every surviving
(peptide, strain) pair is exactly the one it had in the source dataset -- no cluster search is
re-run and no peptide changes fold.

Linearity rule (a peptide is kept only when both agree):

1. DBAASP reports no intrachain bond for it. That flag travels with the built dataset as
   ``has_noncanonical`` (``src.soamp.data.dev_dataset`` writes ``not is_linear`` into it, and
   ``is_linear`` is ``not peptide["intrachainBonds"]`` from the DBAASP record), so no DBAASP
   pull is needed here.
2. Its SMILES has no macrocycle (no ring of 9+ atoms: amino-acid side-chain rings -- Trp, His,
   Phe, Tyr, Pro -- are all 5- or 6-membered) and no disulfide bond (no S-S).

Rule 1 alone would keep 4 peptides out of 8,478 whose generated SMILES does carry a macrocycle
or an S-S bond; rule 2 alone would keep 25 whose SMILES looks linear although DBAASP records a
bond (the SMILES generator falls back to the linear backbone when it cannot place the bond).
Requiring both drops those 29 ambiguous peptides, so every peptide kept is linear under both
the database annotation and the structure the model actually sees. The DBAASP flag is the primary
source and rule 2 only cross-checks it: the four peptides the cross-check removes carry a
tert-butyl disulfide protecting group or a lipoic-acid cap (sulfur in a side chain, not a backbone
bridge), 5 pairs out of 12,722, so a dataset built from rule 1 alone would differ far below fold
noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from soamp.utils.config import REPO_ROOT

#: A ring this large cannot be an amino-acid side chain, so it is a cyclization of the backbone.
MACROCYCLE_MIN_RING_SIZE = 9

#: Two-connected sulfur bonded to two-connected sulfur: a disulfide bridge.
DISULFIDE_SMARTS = "[#16X2][#16X2]"

LINEARITY_RULE = (
    f"kept when DBAASP records no intrachain bond (has_noncanonical == False) AND the SMILES has no "
    f"ring of {MACROCYCLE_MIN_RING_SIZE}+ atoms and no disulfide ({DISULFIDE_SMARTS})"
)


def smiles_is_linear(smiles: str) -> bool | None:
    """False when the SMILES carries a macrocycle or a disulfide, None when it cannot be parsed."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if any(len(ring) >= MACROCYCLE_MIN_RING_SIZE for ring in mol.GetRingInfo().AtomRings()):
        return False
    return not mol.HasSubstructMatch(Chem.MolFromSmarts(DISULFIDE_SMARTS))


def peptide_linearity(dataset: pd.DataFrame) -> pd.DataFrame:
    """One row per peptide of ``dataset``: the two flags and the reason it is dropped (``""`` = kept).

    ``dataset`` is a built ``mic_classification_dataset.csv`` (columns ``peptide_id``, ``smiles``,
    ``has_noncanonical``).
    """
    peptides = dataset.drop_duplicates("peptide_id")[["peptide_id", "smiles", "has_noncanonical"]].copy()
    peptides["dbaasp_linear"] = ~peptides["has_noncanonical"].astype(bool)
    peptides["smiles_linear"] = [smiles_is_linear(s) for s in peptides["smiles"]]

    def reason(row: pd.Series) -> str:
        if row["smiles_linear"] is None:
            return "smiles_unparsable"
        if not row["dbaasp_linear"] and not row["smiles_linear"]:
            return "intrachain_bond_and_cyclic_smiles"
        if not row["dbaasp_linear"]:
            return "dbaasp_intrachain_bond"
        return "" if row["smiles_linear"] else "cyclic_or_disulfide_smiles"

    peptides["drop_reason"] = peptides.apply(reason, axis=1)
    peptides["linear"] = peptides["drop_reason"].eq("")
    return peptides.drop(columns="smiles").reset_index(drop=True)


@dataclass(frozen=True)
class LinearSubset:
    dataset: pd.DataFrame        # the source dataset's rows whose peptide is linear
    peptide_split: pd.DataFrame  # the source peptide_split rows of those peptides
    linearity: pd.DataFrame      # per-peptide flags and drop reason (audit file)
    per_strain: pd.DataFrame     # kept/dropped pairs and active share per strain
    stats: dict


def _per_strain(dataset: pd.DataFrame, linear_ids: set) -> pd.DataFrame:
    d = dataset.assign(kept=dataset["peptide_id"].isin(linear_ids), active=dataset["label"].eq("active"))
    out = pd.DataFrame({
        "pairs_before": d.groupby("organism").size(),
        "pairs_kept": d[d["kept"]].groupby("organism").size(),
        "active_before": d[d["active"]].groupby("organism").size(),
        "active_kept": d[d["kept"] & d["active"]].groupby("organism").size(),
    }).fillna(0).astype(int)
    out["pairs_dropped"] = out["pairs_before"] - out["pairs_kept"]
    out["dropped_%"] = (100 * out["pairs_dropped"] / out["pairs_before"]).round(1)
    out["active_%_before"] = (100 * out["active_before"] / out["pairs_before"]).round(1)
    out["active_%_kept"] = (100 * out["active_kept"] / out["pairs_kept"].where(out["pairs_kept"] > 0)).round(1)
    return out[["pairs_before", "pairs_kept", "pairs_dropped", "dropped_%", "active_%_before", "active_%_kept"]]


def linear_subset(dataset: pd.DataFrame, peptide_split: pd.DataFrame) -> LinearSubset:
    """Filter a built dataset and its peptide split to the linear peptides (see module docstring)."""
    linearity = peptide_linearity(dataset)
    linear_ids = set(linearity.loc[linearity["linear"], "peptide_id"])

    kept = dataset[dataset["peptide_id"].isin(linear_ids)].reset_index(drop=True)
    kept_split = peptide_split[peptide_split["peptide_id"].isin(linear_ids)].reset_index(drop=True)
    per_strain = _per_strain(dataset, linear_ids)

    def by_split(df: pd.DataFrame) -> dict:
        return dict(sorted(df["split"].value_counts().items()))

    stats = {
        "rule": LINEARITY_RULE,
        "peptides_before": int(dataset["peptide_id"].nunique()),
        "peptides_kept": len(linear_ids),
        "peptides_dropped": dict(sorted(linearity.loc[~linearity["linear"], "drop_reason"].value_counts().items())),
        "pairs_before": len(dataset),
        "pairs_kept": len(kept),
        "labels_before": dict(sorted(dataset["label"].value_counts().items())),
        "labels_kept": dict(sorted(kept["label"].value_counts().items())),
        "pairs_by_split_before": by_split(dataset),
        "pairs_by_split_kept": by_split(kept),
        "folds_kept": dict(sorted(kept_split["fold_id"].dropna().astype(int).value_counts().items())),
    }
    return LinearSubset(dataset=kept, peptide_split=kept_split, linearity=linearity,
                        per_strain=per_strain, stats=stats)


def subset_feature_csv(features: pd.DataFrame, linear_ids: set) -> pd.DataFrame:
    """Rows of a ``peptide_features_*.csv`` whose peptide survives the filter."""
    return features[features["peptide_id"].isin(linear_ids)].reset_index(drop=True)


class LinearSubsetPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_dir: Path = Path("data/dev_v2")
    out_dir: Path = Path("data/dev_v2_linear")
    reports_dir: Path = Path("reports/dev_v2_linear")


class LinearSubsetConfig(BaseModel):
    """Schema for ``config/data/dev_v2_linear.yaml`` (pipeline/data/05_build_linear_subset.py)."""

    model_config = ConfigDict(extra="forbid")

    paths: LinearSubsetPaths = LinearSubsetPaths()
    dataset_filename: str = "mic_classification_dataset.csv"
    peptide_split_filename: str = "peptide_split.csv"
    # Per-peptide feature tables of the source dataset, filtered to the kept peptides (absent ones
    # are skipped: the scaler stage and the data artifact say which ones a run actually needs).
    feature_files: list[str] = ["peptide_features_rdkit.csv", "peptide_features_peptideclm.csv"]
    # Files copied through unchanged (the organism side does not depend on the peptides).
    copy_files: list[str] = ["organism_vocab_vocab_embedding.json"]
    linearity_filename: str = "peptide_linearity.csv"
    per_strain_filename: str = "linear_subset_per_strain.csv"
    summary_filename: str = "build_linear_subset_log.txt"

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else REPO_ROOT / p
