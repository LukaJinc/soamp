"""Pure analysis of cross-validation results -- no wandb, matplotlib or LaTeX.

Input is the row-level `cv_results` table pipeline/train_cv.py writes
(reports/cv_results_<exp_id>.csv, also logged as a wandb Artifact): one row
per (dataset row x eval_group) per CV round, with `eval_group` in {"fit",
"val"}, `val_fold_id` (which round), `true_label_int`, `logit` and
`has_noncanonical`. Everything here derives from that one table, so it works
for every run version regardless of how the run logged its history.
"""
import pandas as pd

from soamp.engine.metrics import compute_binary_metrics

PEPTIDE_NAMES = {
    "rdkit": "RDKit",
    "peptideclm": "PeptideCLM",
    "molgraph": "GNN",
}
ORGANISM_NAMES = {
    "vocab": "vocab",
    "kmer": "k-mer",
    "dnabert": "DNABERT-S",
}

REQUIRED_COLUMNS = {"eval_group", "val_fold_id", "true_label_int", "logit", "has_noncanonical"}


class CVAnalysisError(ValueError):
    """Raised when a cv_results table lacks required columns or a slice the
    requested analysis needs."""


def _validate(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise CVAnalysisError(f"cv_results is missing columns: {sorted(missing)}")


def pretty_label(exp_id: str) -> str:
    """'rdkit_vocab_cv' -> 'RDKit x vocab' via the <peptide>_<organism>_cv
    naming convention; an id that doesn't follow it is returned unchanged."""
    parts = exp_id.removesuffix("_cv").split("_")
    if len(parts) == 2 and parts[0] in PEPTIDE_NAMES and parts[1] in ORGANISM_NAMES:
        return f"{PEPTIDE_NAMES[parts[0]]} × {ORGANISM_NAMES[parts[1]]}"
    return exp_id


def per_fold_fit_val(cv_results: pd.DataFrame) -> pd.DataFrame:
    """One row per (val_fold_id, eval_group) with the binary metrics of that
    round: columns fold, eval_group, accuracy, f1, precision, recall, auroc.
    `fit` = the four training folds of the round, `val` = the held-out fold."""
    _validate(cv_results)
    rows = []
    for (fold, group), part in cv_results.groupby(["val_fold_id", "eval_group"]):
        metrics = compute_binary_metrics(
            part["logit"].to_numpy(), part["true_label_int"].to_numpy()
        )
        rows.append({"fold": int(fold), "eval_group": str(group), **metrics})
    return pd.DataFrame(rows).sort_values(["eval_group", "fold"]).reset_index(drop=True)


def noncanonical_recall(cv_results: pd.DataFrame) -> dict[str, float]:
    """Pooled out-of-fold recall (positive class = active) on canonical vs
    non-canonical peptides: every row is held out in exactly one round, so
    pooling the `val` rows gives one prediction per (peptide, organism) pair.
    Returns canonical_recall, noncanonical_recall, delta (non-canonical minus
    canonical), n_canonical, n_noncanonical."""
    _validate(cv_results)
    val = cv_results[cv_results["eval_group"] == "val"]
    out: dict[str, float] = {}
    for is_nc, name in ((False, "canonical"), (True, "noncanonical")):
        part = val[val["has_noncanonical"].astype(bool) == is_nc]
        if part.empty or part["true_label_int"].nunique() < 2:
            raise CVAnalysisError(f"no usable {name} slice in the val rows")
        out[f"{name}_recall"] = compute_binary_metrics(
            part["logit"].to_numpy(), part["true_label_int"].to_numpy()
        )["recall"]
        out[f"n_{name}"] = int(len(part))
    out["delta"] = out["noncanonical_recall"] - out["canonical_recall"]
    return out


def noncanonical_recall_table(results_by_run: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per run (index = run id, order preserved): canonical_recall,
    noncanonical_recall, delta, n_canonical, n_noncanonical."""
    return pd.DataFrame(
        {run: noncanonical_recall(df) for run, df in results_by_run.items()}
    ).T
