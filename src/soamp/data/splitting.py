"""Pure train/validation carve-out.

Splits mic_classification_dataset.csv's 'train' rows into a fitting subset
and a held-out validation subset used only for checkpoint 'best' selection
and per-epoch monitoring -- the true 'test' split is never touched here.
"""
import random


class SplitError(ValueError):
    """Raised when val_fraction is outside (0, 1), or when it rounds down
    to 0 validation peptide_ids."""


def split_train_validation(
    train_rows: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Splits by unique peptide_id (not by row) -- a peptide's rows across
    multiple organisms must land in the same bucket, or a model could see
    a peptide's E. coli row in fit and its S. aureus row in val. Shuffled
    deterministically via random.Random(seed). Returns (fit_rows, val_rows)."""
    if not (0.0 < val_fraction < 1.0):
        raise SplitError(f"val_fraction must be in (0, 1), got {val_fraction}")

    peptide_ids = sorted({row["peptide_id"] for row in train_rows})
    n_val = round(len(peptide_ids) * val_fraction)
    if n_val == 0:
        raise SplitError(
            f"val_fraction={val_fraction} rounds down to 0 validation "
            f"peptide_ids out of {len(peptide_ids)} total"
        )

    rng = random.Random(seed)
    shuffled = list(peptide_ids)
    rng.shuffle(shuffled)
    val_ids = set(shuffled[:n_val])

    fit_rows = [r for r in train_rows if r["peptide_id"] not in val_ids]
    val_rows = [r for r in train_rows if r["peptide_id"] in val_ids]
    return fit_rows, val_rows
