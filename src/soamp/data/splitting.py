"""Homology-aware train/validation carve-out.

Splits mic_classification_dataset.csv's 'train' rows into a fitting subset
and a held-out validation subset used only for checkpoint 'best' selection
and per-epoch monitoring -- the true 'test' split is never touched here.

Uses the same mechanism as the train/test split
(pipeline/curation/07_build_final_and_split.py): qmap.toolkit.train_test_split
-- a sequence-identity graph + Leiden community detection split -- so that
two homologous peptides can never land one in 'fit' and one in 'val'. A
plain random split (the previous implementation) would let that happen,
optimistically biasing val_auroc relative to true generalization.

qmap.toolkit.train_test_split defaults to post_filtering=True: after
clustering, it removes any peptide assigned to the first ('fit') side that
has a similarity edge (>= identity_threshold) to a peptide on the second
('val') side, to guarantee independence -- and drops those peptides from
its return value entirely, with no record of which were removed. This
mirrors 07_build_final_and_split.py's reconciliation: anything missing
from both returned sets is reassigned to 'val', never back to 'fit', since
the filter only ever flags a peptide for being *too close to val*.
"""


class SplitError(ValueError):
    """Raised when val_fraction is outside (0, 1), or when it rounds down
    to 0 validation peptide_ids."""


def apply_val_split(
    train_rows: list[dict], val_peptide_ids: set[str]
) -> tuple[list[dict], list[dict]]:
    """Partitions train_rows by a precomputed val_peptide_ids set (e.g.
    from data/val_split.json) -- the "apply an already-computed split"
    counterpart to split_train_validation, which computes one. Returns
    (fit_rows, val_rows)."""
    fit_rows = [r for r in train_rows if r["peptide_id"] not in val_peptide_ids]
    val_rows = [r for r in train_rows if r["peptide_id"] in val_peptide_ids]
    return fit_rows, val_rows


def split_train_validation(
    train_rows: list[dict],
    val_fraction: float,
    seed: int,
    identity_threshold: float = 0.60,
    post_filtering: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Splits by unique peptide_id (not row) -- a peptide's rows across
    multiple organisms must land in the same bucket, or a model could see
    a peptide's E. coli row in fit and its S. aureus row in val.

    The split itself is computed at the unique-peptide level via
    qmap.toolkit.train_test_split on each peptide's sequence, deterministic
    given seed. Returns (fit_rows, val_rows)."""
    if not (0.0 < val_fraction < 1.0):
        raise SplitError(f"val_fraction must be in (0, 1), got {val_fraction}")

    unique_peptides: dict[str, str] = {}
    for row in train_rows:
        pid = row["peptide_id"]
        if pid not in unique_peptides:
            unique_peptides[pid] = row["sequence"]
    peptide_ids = list(unique_peptides.keys())
    sequences = [unique_peptides[pid] for pid in peptide_ids]

    n_val = round(len(peptide_ids) * val_fraction)
    if n_val == 0:
        raise SplitError(
            f"val_fraction={val_fraction} rounds down to 0 validation "
            f"peptide_ids out of {len(peptide_ids)} total"
        )

    from qmap.toolkit import train_test_split

    fit_seqs, val_seqs, fit_ids, val_ids = train_test_split(
        sequences, peptide_ids,
        threshold=identity_threshold, test_size=val_fraction,
        random_state=seed, post_filtering=post_filtering, verbose=True,
    )

    # post_filtering (default True) removes fit peptides with an edge to a
    # val peptide, and drops them from both returned lists with no record.
    # Reconcile against the full peptide set and reassign anything missing
    # to val -- not fit, since the filter only flags a peptide for being
    # too close to val.
    assigned = set(fit_ids) | set(val_ids)
    dropped_ids = sorted(set(peptide_ids) - assigned)
    val_ids = list(val_ids) + dropped_ids

    val_id_set = set(val_ids)
    fit_rows = [r for r in train_rows if r["peptide_id"] not in val_id_set]
    val_rows = [r for r in train_rows if r["peptide_id"] in val_id_set]
    return fit_rows, val_rows
