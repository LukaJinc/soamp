"""Build the classification dataset + CV split from the "dev" DBAASP data.

Inputs are the artifacts of scripts/EDA/09_data_processing.ipynb:
``data/dev/dbaasp_dev_processed.parquet`` (one row per peptide x target
activity record: ``activity`` in uM, ``concentration`` raw string, ``measure``)
and ``data/dev/dbaasp_dev_split.parquet`` (test + 5 CV folds per peptide).
Outputs mirror the shapes the existing training pipeline reads
(``mic_classification_dataset.csv`` + ``peptide_split.csv``), so training runs
only need ``paths.data_dir: data/dev``.

Labeling reuses the shared threshold table (soamp.common.thresholds) and
``derive_label`` (soamp.labeling.labels). Pure logic only -- I/O lives in
pipeline/data/03_build_dev_classification_dataset.py.

UNITS: ``activity`` in the dev data is DBAASP's value normalised to **ug/ml**
(rows reported in uM were multiplied by MW/1000; ug/ml rows are unchanged), not
uM. By decision, the threshold table's 32 / 128 are applied to ``activity``
*as stored*, with no MW conversion. The output column ``mic_value_uM`` (legacy
name, kept so downstream readers don't change) therefore holds that same
ug/ml-normalised number; ``THRESHOLD_UNIT`` names it for logging.

Dev v2 (``value_column: mic_ug_ml``): the value is not DBAASP's ``activity`` but the
MIC recomputed from ``concentration`` (soamp.data.mic_recompute), because DBAASP's
molecular weight is wrong for some peptides. With ``include_na_rows`` the DBAASP
"not active" rows (concentration ``NA``, sentinel 1,111,111, measure '-' or empty)
are admitted next to the MIC rows; at 1,111,111 they are inactive under any cutoff.

Label rules, per record (the value, unit above):
- exact value ``a``      -> interval [a, a]
- ``>a`` / ``>=a``       -> true MIC in [a, inf)  (right-censored)
- ``<a`` / ``<=a``       -> true MIC in [0, a]    (left-censored)
  (disable with ``use_censoring=False`` to treat the value as exact.)
Active only if the whole interval is <= active threshold, inactive only if
it is >= inactive threshold, otherwise uncertain (dropped). Records of the
same (peptide, organism) are then merged: agreeing labels keep that label,
disagreeing ones are dropped as conflicting.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from soamp.common.thresholds import ThresholdTable, lookup_threshold, species_of
from soamp.data.mic_recompute import NA_SENTINEL, parse_not_active_limit
from soamp.labeling.labels import derive_label
from soamp.splitting.config import BucketingConfig
from soamp.utils.config import REPO_ROOT

DATASET_COLUMNS = [
    "peptide_id", "sequence", "smiles", "organism", "ncbi_taxon_id_if_available",
    "mic_value_uM", "mic_type", "has_noncanonical", "label", "split",
]
THRESHOLD_UNIT = "µg/ml (DBAASP activity, unconverted)"
THRESHOLD_UNIT_RECOMPUTED = "µg/ml (recomputed from concentration)"
SPLIT_COLUMNS = ["peptide_id", "community", "split", "fold_id", "has_noncanonical"]


class DevDatasetError(ValueError):
    pass


class DevPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processed_parquet: Path = Path("data/dev/dbaasp_dev_processed.parquet")
    # Old configs: INPUT, a ready-made split per peptide. When ``cluster_parquet`` is set the
    # split is computed from the clusters after labelling and written to this path instead.
    split_parquet: Path = Path("data/dev/dbaasp_dev_split.parquet")
    cluster_parquet: Path | None = None   # peptide_id, cluster_id, cluster_kind (e.g. dbaasp_dev_split_v2.parquet)
    thresholds_csv: Path = Path("config/thresholds/dev_dbaasp_thresholds.csv")
    out_dir: Path = Path("data/dev")
    reports_dir: Path = Path("reports")


class DevDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: DevPaths = DevPaths()
    target_organisms: list[str] = [
        "Escherichia coli ATCC 25922",
        "Staphylococcus aureus ATCC 25923",
        "Pseudomonas aeruginosa ATCC 27853",
    ]
    measure: str = "MIC"
    use_censoring: bool = True
    # True: "active" means MIC strictly below the active threshold and "inactive"
    # strictly above the inactive one (DBAASP's <25 / >100). False: inclusive.
    strict_boundaries: bool = True
    # Column holding the value the cutoffs apply to: DBAASP's ``activity`` (old behaviour) or
    # the recomputed ``mic_ug_ml`` (soamp.data.mic_recompute).
    value_column: str = "activity"
    # Column holding gt/lt/exact; None -> derived from the raw ``concentration`` text.
    direction_column: str | None = None
    # Also admit rows whose concentration is NA (DBAASP "not active"), whatever their measure.
    include_na_rows: bool = False
    # Bucket search used when ``paths.cluster_parquet`` is set (production defaults).
    bucketing: BucketingConfig = BucketingConfig()
    dataset_filename: str = "mic_classification_dataset.csv"
    peptide_split_filename: str = "peptide_split.csv"
    split_sidecar_filename: str = "dev_split_final.json"
    na_only_filename: str = "na_only_pairs.csv"
    conflicts_filename: str = "label_conflicts.csv"
    summary_filename: str = "build_dev_classification_dataset_log.txt"

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else REPO_ROOT / p


def censor_direction(concentration: object) -> str:
    """'gt' for '>x'/'>=x'/'≥x', 'lt' for '<x'/'<=x'/'≤x', else 'exact'."""
    s = str(concentration).strip()
    if s.startswith((">", "≥")):
        return "gt"
    if s.startswith(("<", "≤")):
        return "lt"
    return "exact"


def label_record(activity: float, direction: str, active_uM: float | None, inactive_uM: float | None,
                 use_censoring: bool = True) -> str:
    if direction == "exact" or not use_censoring:
        label, _ = derive_label(activity, "exact", active_uM, inactive_uM)
    elif direction == "gt":
        label, _ = derive_label(activity, "censored", active_uM, inactive_uM, activity, math.inf)
    else:
        label, _ = derive_label(activity, "censored", active_uM, inactive_uM, 0.0, activity)
    return label


def merge_labels(labels: list[str]) -> str | None:
    """None when there is no decisive record or the decisive ones disagree."""
    decisive = {lab for lab in labels if lab in ("active", "inactive")}
    return decisive.pop() if len(decisive) == 1 else None


def na_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose concentration is DBAASP's 'NA' (no activity measured / not active)."""
    if "conc_kind" in df.columns:
        return df["conc_kind"].eq("NA")
    return df["concentration"].astype(str).str.strip().eq("NA")


@dataclass(frozen=True)
class LabelResult:
    dataset: pd.DataFrame      # DATASET_COLUMNS without ``split``: one row per labelled (peptide, organism)
    stats: dict
    na_only: pd.DataFrame      # labelled pairs that rest only on NA rows (audit)
    conflicts: pd.DataFrame    # pairs dropped because decisive record labels disagreed (audit)


@dataclass(frozen=True)
class BuildResult:
    dataset: pd.DataFrame
    peptide_split: pd.DataFrame
    stats: dict


def label_dev_pairs(processed: pd.DataFrame, table: ThresholdTable, cfg: DevDatasetConfig) -> LabelResult:
    """Record labels -> one label per (peptide, organism). No split is needed or attached."""
    in_scope = processed["target_species"].isin(cfg.target_organisms)
    keep = in_scope & (processed["measure"] == cfg.measure)
    if cfg.include_na_rows:
        keep |= in_scope & na_mask(processed)
    rows = processed[keep]
    rows = rows[rows[cfg.value_column].notna()].copy()
    if rows.empty:
        raise DevDatasetError("no rows left after organism/measure filtering")
    rows["_value"] = rows[cfg.value_column].astype(float)
    rows["_is_na"] = na_mask(rows) if cfg.include_na_rows else False

    thresholds = {o: lookup_threshold(table, species_of(o)) for o in cfg.target_organisms}
    no_thr = [o for o, m in thresholds.items() if m is None]
    if no_thr:
        raise DevDatasetError(f"no threshold for organisms {no_thr} in the threshold table")

    # derive_label is inclusive (<= active, >= inactive); nudging the cutoffs by one
    # float step makes the comparisons strict without touching the shared function.
    cutoffs = {}
    for o, match in thresholds.items():
        act, inact = match.active_threshold_uM, match.inactive_threshold_uM
        if cfg.strict_boundaries:
            act, inact = math.nextafter(act, -math.inf), math.nextafter(inact, math.inf)
        cutoffs[o] = (act, inact)

    rows["direction"] = (rows[cfg.direction_column] if cfg.direction_column
                         else rows["concentration"].map(censor_direction))
    rows["record_label"] = [
        label_record(a, d, *cutoffs[o], cfg.use_censoring)
        for a, d, o in zip(rows["_value"], rows["direction"], rows["target_species"])
    ]

    out, na_only, conflicts = [], [], []
    for (pid, org), g in rows.groupby(["peptide_id", "target_species"], sort=True):
        decisive = {lab for lab in g["record_label"] if lab in ("active", "inactive")}
        label = merge_labels(g["record_label"].tolist())
        if label is None:
            if len(decisive) == 2:
                conflicts.append({"peptide_id": pid, "organism": org, "records": len(g),
                                  "involves_na_row": bool(g["_is_na"].any()),
                                  "labels": ",".join(sorted(g["record_label"]))})
            continue
        first = g.iloc[0]
        measured = g[~g["_is_na"]]
        positive = measured["_value"][measured["_value"] > 0]
        all_na = bool(g["_is_na"].all())
        if all_na:
            mic_value, mic_type = NA_SENTINEL, "na"
            note = " | ".join(sorted({str(n) for n in g["note"].dropna() if str(n)})) if "note" in g else ""
            mw = float(g["mw"].iloc[0]) if "mw" in g else float("nan")
            limits = [parse_not_active_limit(n, mw) for n in g["note"]] if "note" in g else []
            limits = [x for x in limits if not np.isnan(x)]
            na_only.append({"peptide_id": pid, "organism": org, "label": label, "records": len(g),
                            "note": note, "not_active_up_to_ug_ml": max(limits) if limits else float("nan")})
        else:
            mic_value = float(math.exp(sum(map(math.log, positive)) / len(positive))) if len(positive) else 0.0
            mic_type = "censored" if (measured["direction"] != "exact").all() else (
                "averaged" if len(measured) > 1 else "exact")
        out.append({
            "peptide_id": pid,
            "sequence": first["sequence_full"],
            "smiles": first["smiles"],
            "organism": org,
            "ncbi_taxon_id_if_available": "",
            "mic_value_uM": mic_value,
            "mic_type": mic_type,
            "has_noncanonical": not bool(first["is_linear"]),
            "label": label,
        })
    if not out:
        raise DevDatasetError("no (peptide, organism) pair received an active/inactive label")
    dataset = pd.DataFrame(out)

    n_dir = rows["direction"].value_counts().to_dict()
    flipped = int(((rows["direction"] == "gt") & (rows["record_label"] != "active") &
                   (rows["_value"] <= min(c[0] for c in cutoffs.values()))).sum())
    stats = {
        "threshold_unit": THRESHOLD_UNIT if cfg.value_column == "activity" else THRESHOLD_UNIT_RECOMPUTED,
        "records": len(rows),
        "na_records": int(rows["_is_na"].sum()),
        "record_labels": rows["record_label"].value_counts().to_dict(),
        "censor_directions": n_dir,
        "gt_records_below_active_threshold_not_labelled_active": flipped,
        "peptide_organism_pairs": rows.groupby(["peptide_id", "target_species"]).ngroups,
        "labelled_pairs": len(dataset),
        "labels": dataset["label"].value_counts().to_dict(),
        "na_only_labelled_pairs": len(na_only),
        "pairs_dropped_for_conflicting_labels": len(conflicts),
        "of_which_involving_an_na_row": int(sum(c["involves_na_row"] for c in conflicts)),
        "peptides": int(dataset["peptide_id"].nunique()),
        "by_organism": dataset.groupby("organism")["label"].value_counts().unstack(fill_value=0).to_dict("index"),
    }
    return LabelResult(dataset, stats, pd.DataFrame(na_only), pd.DataFrame(conflicts))


def attach_split(dataset: pd.DataFrame, split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the ``split`` column and build the per-peptide split table the training code reads."""
    split = split.set_index("peptide_id")
    missing = sorted(set(dataset["peptide_id"]) - set(split.index))
    if missing:
        raise DevDatasetError(f"{len(missing)} peptides have no split entry, e.g. {missing[:5]}")
    dataset = dataset.copy()
    dataset["split"] = dataset["peptide_id"].map(split["split"])
    dataset = dataset[DATASET_COLUMNS]

    pep = dataset.drop_duplicates("peptide_id").set_index("peptide_id")
    ps = split.loc[pep.index]
    peptide_split = pd.DataFrame({
        "peptide_id": pep.index,
        "community": ps["cluster_id"].values,
        "split": ps["split"].values,
        "fold_id": [("" if pd.isna(f) else str(int(f))) for f in ps["fold_id"]],
        "has_noncanonical": pep["has_noncanonical"].values,
    })[SPLIT_COLUMNS]
    return dataset, peptide_split


def build_dev_dataset(processed: pd.DataFrame, split: pd.DataFrame, table: ThresholdTable,
                      cfg: DevDatasetConfig) -> BuildResult:
    """Label the pairs and attach a ready-made split (the original behaviour)."""
    labelled = label_dev_pairs(processed, table, cfg)
    dataset, peptide_split = attach_split(labelled.dataset, split)
    stats = {**labelled.stats, "by_split": dataset["split"].value_counts().to_dict()}
    return BuildResult(dataset, peptide_split, stats)
