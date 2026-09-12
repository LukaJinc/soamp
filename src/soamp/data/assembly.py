"""Pure join/filter/split-attachment logic for assembling the
classification-ready dataset from the labeling + curation outputs.

No I/O here -- see pipeline/data/01_build_classification_dataset.py for the
orchestration that reads final_mic_regression_dataset.csv,
mic_activity_labels.csv, and split_indices.json and calls into this module.
"""
from typing import Iterable

FIELDNAMES = [
    "peptide_id", "sequence", "smiles", "organism",
    "ncbi_taxon_id_if_available", "mic_value_uM", "mic_type",
    "has_noncanonical", "label", "split",
]


class DatasetAssemblyError(ValueError):
    """Raised when an invariant this stage relies on doesn't hold: every
    filtered label row's (peptide_id, organism) must have a matching
    final_mic_regression_dataset.csv row, and every peptide_id must have
    exactly one split assignment (not zero, not both)."""


def filter_labels(label_rows: list[dict], included_labels: list[str]) -> list[dict]:
    """Keep only rows whose 'label' is in included_labels (e.g. drops
    'uncertain'/'unlabeled'). Order-preserving, no dedup."""
    included = set(included_labels)
    return [row for row in label_rows if row["label"] in included]


def join_regression_and_labels(
    regression_rows: list[dict],
    filtered_label_rows: list[dict],
) -> list[dict]:
    """Join filtered_label_rows against regression_rows on
    (peptide_id, organism) -- unique in both files. Pulls sequence/smiles/
    ncbi_taxon_id_if_available/mic_value_uM/mic_type/has_noncanonical from
    the regression row, label from the label row.

    Raises DatasetAssemblyError if a filtered label row's key has no
    match in regression_rows -- the two files are expected to mirror
    each other 1:1 row-for-row, so a miss means the inputs are out of
    sync, not a case to silently skip.
    """
    index = {(r["peptide_id"], r["organism"]): r for r in regression_rows}
    out = []
    for label_row in filtered_label_rows:
        key = (label_row["peptide_id"], label_row["organism"])
        reg_row = index.get(key)
        if reg_row is None:
            raise DatasetAssemblyError(
                f"No final_mic_regression_dataset.csv row for labeled pair "
                f"{key} -- inputs are out of sync, re-run the labeling stage"
            )
        out.append({
            "peptide_id": key[0],
            "sequence": reg_row["sequence"],
            "smiles": reg_row["smiles"],
            "organism": key[1],
            "ncbi_taxon_id_if_available": reg_row["ncbi_taxon_id_if_available"],
            "mic_value_uM": reg_row["mic_value_uM"],
            "mic_type": reg_row["mic_type"],
            "has_noncanonical": reg_row["has_noncanonical"],
            "label": label_row["label"],
        })
    return out


def attach_split(
    rows: list[dict],
    train_peptide_ids: Iterable[str],
    test_peptide_ids: Iterable[str],
) -> list[dict]:
    """Attach a 'split' column ('train'/'test') per row's peptide_id.

    Raises DatasetAssemblyError if a peptide_id is in neither set (no
    split assignment) or in both (train/test overlap) -- both are
    invariant violations of split_indices.json, not cases to guess
    through.
    """
    train_ids = set(train_peptide_ids)
    test_ids = set(test_peptide_ids)
    overlap = train_ids & test_ids
    if overlap:
        raise DatasetAssemblyError(
            f"{len(overlap)} peptide_id(s) in both train and test split "
            f"sets, e.g. {sorted(overlap)[:5]}"
        )
    out = []
    for row in rows:
        pid = row["peptide_id"]
        if pid in train_ids:
            split = "train"
        elif pid in test_ids:
            split = "test"
        else:
            raise DatasetAssemblyError(
                f"peptide_id {pid!r} has no split assignment in split_indices.json"
            )
        out.append({**row, "split": split})
    return out


def build_classification_dataset(
    regression_rows: list[dict],
    label_rows: list[dict],
    train_peptide_ids: Iterable[str],
    test_peptide_ids: Iterable[str],
    included_labels: list[str],
) -> list[dict]:
    """Top-level composition: filter labels -> join -> attach split.
    Returns rows with exactly FIELDNAMES' keys, ready to write as the
    classification-ready CSV."""
    filtered = filter_labels(label_rows, included_labels)
    joined = join_regression_and_labels(regression_rows, filtered)
    return attach_split(joined, train_peptide_ids, test_peptide_ids)
