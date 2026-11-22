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

Label rules, per record (``activity``, unit above):
- exact value ``a``      -> interval [a, a]
- ``>a`` / ``>=a``       -> true MIC in [a, inf)  (right-censored)
- ``<a`` / ``<=a``       -> true MIC in [0, a]    (left-censored)
  (disable with ``use_censoring=False`` to treat ``activity`` as exact.)
Active only if the whole interval is <= active threshold, inactive only if
it is >= inactive threshold, otherwise uncertain (dropped). Records of the
same (peptide, organism) are then merged: agreeing labels keep that label,
disagreeing ones are dropped as conflicting.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from soamp.common.thresholds import ThresholdTable, lookup_threshold, species_of
from soamp.labeling.labels import derive_label
from soamp.utils.config import REPO_ROOT

DATASET_COLUMNS = [
    "peptide_id", "sequence", "smiles", "organism", "ncbi_taxon_id_if_available",
    "mic_value_uM", "mic_type", "has_noncanonical", "label", "split",
]
THRESHOLD_UNIT = "µg/ml (DBAASP activity, unconverted)"
SPLIT_COLUMNS = ["peptide_id", "community", "split", "fold_id", "has_noncanonical"]


class DevDatasetError(ValueError):
    pass


class DevPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processed_parquet: Path = Path("data/dev/dbaasp_dev_processed.parquet")
    split_parquet: Path = Path("data/dev/dbaasp_dev_split.parquet")
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
    dataset_filename: str = "mic_classification_dataset.csv"
    peptide_split_filename: str = "peptide_split.csv"
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


@dataclass(frozen=True)
class BuildResult:
    dataset: pd.DataFrame
    peptide_split: pd.DataFrame
    stats: dict


def build_dev_dataset(processed: pd.DataFrame, split: pd.DataFrame, table: ThresholdTable,
                      cfg: DevDatasetConfig) -> BuildResult:
    rows = processed[processed["target_species"].isin(cfg.target_organisms) & (processed["measure"] == cfg.measure)]
    rows = rows[rows["activity"].notna()].copy()
    if rows.empty:
        raise DevDatasetError("no rows left after organism/measure filtering")

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

    rows["direction"] = rows["concentration"].map(censor_direction)
    rows["record_label"] = [
        label_record(a, d, *cutoffs[o], cfg.use_censoring)
        for a, d, o in zip(rows["activity"], rows["direction"], rows["target_species"])
    ]

    out = []
    for (pid, org), g in rows.groupby(["peptide_id", "target_species"], sort=True):
        label = merge_labels(g["record_label"].tolist())
        if label is None:
            continue
        first = g.iloc[0]
        positive = g["activity"][g["activity"] > 0]
        out.append({
            "peptide_id": pid,
            "sequence": first["sequence_full"],
            "smiles": first["smiles"],
            "organism": org,
            "ncbi_taxon_id_if_available": "",
            "mic_value_uM": float(math.exp(sum(map(math.log, positive)) / len(positive))) if len(positive) else 0.0,
            "mic_type": "censored" if (g["direction"] != "exact").all() else ("averaged" if len(g) > 1 else "exact"),
            "has_noncanonical": not bool(first["is_linear"]),
            "label": label,
        })
    if not out:
        raise DevDatasetError("no (peptide, organism) pair received an active/inactive label")
    dataset = pd.DataFrame(out)

    split = split.set_index("peptide_id")
    missing = sorted(set(dataset["peptide_id"]) - set(split.index))
    if missing:
        raise DevDatasetError(f"{len(missing)} peptides have no split entry, e.g. {missing[:5]}")
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

    n_dir = rows["direction"].value_counts().to_dict()
    flipped = int(((rows["direction"] == "gt") & (rows["record_label"] != "active") &
                   (rows["activity"] <= min(c[0] for c in cutoffs.values()))).sum())
    stats = {
        "threshold_unit": THRESHOLD_UNIT,
        "records": len(rows),
        "record_labels": rows["record_label"].value_counts().to_dict(),
        "censor_directions": n_dir,
        "gt_records_below_active_threshold_not_labelled_active": flipped,
        "peptide_organism_pairs": rows.groupby(["peptide_id", "target_species"]).ngroups,
        "labelled_pairs": len(dataset),
        "labels": dataset["label"].value_counts().to_dict(),
        "peptides": int(dataset["peptide_id"].nunique()),
        "by_organism": dataset.groupby("organism")["label"].value_counts().unstack(fill_value=0).to_dict("index"),
        "by_split": dataset["split"].value_counts().to_dict(),
    }
    return BuildResult(dataset, peptide_split, stats)
