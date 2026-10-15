"""
Builds the two result visuals every experiment module of the presentation
shows, from a set of cross-validation runs:

  fit_vs_val.png     per-fold fit vs validation precision/recall, one panel
                     per run  -> the overfit / underfit slide
  nc_recall_table.tex  canonical vs non-canonical recall and their delta
                     -> the non-canonical slide (\\input'ed as a tabular)
  meta.json          which runs / source / row counts the assets came from

Output goes to presentations/experiments/<name>/ (override with --out-dir);
the slides pick the files up through \\optfig / \\IfFileExists, so the deck
compiles before they exist.

Source (pick one):
  --wandb-group GROUP   every kfold_cv run of that wandb group (or --runs a,b)
  --csv-dir reports/    reports/cv_results_<run>.csv, which pipeline/train_cv.py
                        writes (and the Colab notebook copies to Drive); needs --runs

Examples:
  python scripts/slides/build_experiment_assets.py --name featurization_grid \\
      --wandb-group featurization_grid_cv_v2
  python scripts/slides/build_experiment_assets.py --name featurization_grid \\
      --csv-dir reports --runs rdkit_vocab_cv,peptideclm_kmer_cv

All analysis lives in soamp.eval.cv_analysis; this script only does I/O,
plotting and LaTeX formatting.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from soamp.eval.cv_analysis import (  # noqa: E402
    noncanonical_recall_table,
    per_fold_fit_val,
    pretty_label,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# presentations/classification_model_v1.tex palette
PAPER, INK, TEAL, RUST, MUTED = "#F6F4EE", "#10222B", "#1F7A6D", "#B4531A", "#5F6B73"

BOOL_COLS = ["has_noncanonical", "correct"]
INT_COLS = ["true_label_int", "pred_label_int", "fold_id", "val_fold_id"]
FLOAT_COLS = ["logit", "prob"]


# ---------------------------------------------------------------- loading
def _to_bool(series: pd.Series) -> pd.Series:
    # wandb Tables reload every column as a string; bool("False") is True, so
    # map the literal strings instead of casting.
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).map({"True": True, "False": False}).astype(bool)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col in BOOL_COLS:
        if col in df:
            df[col] = _to_bool(df[col])
    for col in INT_COLS:
        if col in df:
            df[col] = df[col].astype(int)
    for col in FLOAT_COLS:
        if col in df:
            df[col] = df[col].astype(float)
    return df


def load_from_csv(csv_dir: Path, runs: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for run in runs:
        path = csv_dir / f"cv_results_{run}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (run pipeline/train_cv.py first)")
        out[run] = _coerce(pd.read_csv(path))
    return out


def load_from_wandb(group: str, runs: list[str] | None) -> tuple[dict[str, pd.DataFrame], dict]:
    import wandb
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    found = {
        r.name: r
        for r in api.runs(f"{entity}/soamp", filters={"group": group, "jobType": "kfold_cv"})
        if r.state == "finished"
    }
    names = runs or sorted(found)
    missing = [n for n in names if n not in found]
    if missing:
        raise ValueError(f"no finished kfold_cv run(s) {missing} in group {group!r}; have {sorted(found)}")
    out, created = {}, {}
    for name in names:
        # Each run's OWN artifact: all runs log one artifact name (cv_results),
        # so `cv_results:latest` project-wide would resolve to the last run only.
        artifacts = [a for a in found[name].logged_artifacts() if a.name.startswith("cv_results")]
        if len(artifacts) != 1:
            raise ValueError(f"{name}: expected one cv_results artifact, got {[a.name for a in artifacts]}")
        table = artifacts[0].get("cv_results")
        out[name] = _coerce(pd.DataFrame(table.data, columns=table.columns))
        created[name] = str(found[name].created_at)
    return out, {"wandb_group": group, "run_created_at": created}


# --------------------------------------------------------------- plotting
def plot_fit_vs_val(results: dict[str, pd.DataFrame], out_path: Path) -> None:
    n = len(results)
    ncols = 2 if n <= 4 else 3
    nrows = math.ceil(n / ncols)
    per_fold = {run: per_fold_fit_val(df) for run, df in results.items()}

    lowest = min(
        frame[["precision", "recall"]].to_numpy().min() for frame in per_fold.values()
    )
    y_min = max(0.0, math.floor((lowest - 0.05) * 10) / 10)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.45 * ncols + 0.15, 1.0 * nrows + 0.6),
        sharex=True, sharey=True, squeeze=False, facecolor=PAPER,
    )
    for ax, (run, frame) in zip(axes.flat, per_fold.items()):
        ax.set_facecolor(PAPER)
        for group, style, alpha in (("fit", "-", 1.0), ("val", "--", 0.9)):
            part = frame[frame["eval_group"] == group]
            ax.plot(part["fold"], part["precision"], style, marker="o", ms=3, lw=1.3,
                    color=TEAL, alpha=alpha)
            ax.plot(part["fold"], part["recall"], style, marker="s", ms=3, lw=1.3,
                    color=RUST, alpha=alpha)
        ax.set_title(pretty_label(run), fontsize=8, color=INK, fontweight="bold")
        ax.set_ylim(y_min, 1.0)
        ax.set_xticks(sorted(frame["fold"].unique()))
        ax.grid(alpha=0.35, color=MUTED, lw=0.5)
        ax.tick_params(colors=MUTED, labelsize=6.5, length=2)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("held-out fold", fontsize=6.5, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("precision / recall", fontsize=6.5, color=MUTED)

    legend = [
        Line2D([], [], color=TEAL, marker="o", ms=3, lw=1.3, label="precision"),
        Line2D([], [], color=RUST, marker="s", ms=3, lw=1.3, label="recall"),
        Line2D([], [], color=INK, lw=1.3, ls="-", label="fit (train folds)"),
        Line2D([], [], color=INK, lw=1.3, ls="--", label="val (held-out fold)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=6.5, frameon=False,
               labelcolor=INK, borderaxespad=0.1)
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.5, h_pad=0.6, w_pad=0.6)
    fig.savefig(out_path, dpi=300, facecolor=PAPER)
    plt.close(fig)


# ---------------------------------------------------------------- LaTeX
def _tex_label(run: str) -> str:
    label = pretty_label(run).replace("×", "$\\times$")
    return label.replace("_", "\\_")


def write_nc_table(results: dict[str, pd.DataFrame], out_path: Path) -> pd.DataFrame:
    table = noncanonical_recall_table(results)
    lines = [
        "% generated by scripts/slides/build_experiment_assets.py -- do not edit by hand",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}l c c c@{}}",
        "\\toprule",
        " & \\multicolumn{2}{c}{\\textbf{recall}} & \\\\",
        "\\cmidrule(lr){2-3}",
        " & \\textbf{canonical} & \\textbf{non-canonical} & \\textbf{delta} \\\\",
        "\\midrule",
    ]
    for run, row in table.iterrows():
        # Delta from the DISPLAYED (3-decimal) columns, so the slide adds up
        # for anyone subtracting them; the library's unrounded delta can
        # differ from that by 0.001.
        delta = round(row["noncanonical_recall"], 3) - round(row["canonical_recall"], 3)
        colour = "teal" if delta >= 0 else "rust"
        sign = "+" if delta >= 0 else "$-$"
        lines.append(
            f"{_tex_label(run)} & {row['canonical_recall']:.3f} & "
            f"\\textbf{{{row['noncanonical_recall']:.3f}}} & "
            f"{{\\color{{{colour}}}{sign}{abs(delta):.3f}}} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n")
    return table


# ------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="experiment module name, e.g. featurization_grid")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wandb-group")
    source.add_argument("--csv-dir", type=Path)
    parser.add_argument("--runs", help="comma-separated run ids (required with --csv-dir)")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    runs = [r.strip() for r in args.runs.split(",")] if args.runs else None
    if args.csv_dir is not None:
        if not runs:
            parser.error("--csv-dir needs --runs")
        results = load_from_csv(args.csv_dir, runs)
        source_meta = {"csv_dir": str(args.csv_dir)}
    else:
        results, source_meta = load_from_wandb(args.wandb_group, runs)

    out_dir = args.out_dir or REPO_ROOT / "presentations" / "experiments" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_fit_vs_val(results, out_dir / "fit_vs_val.png")
    nc_table = write_nc_table(results, out_dir / "nc_recall_table.tex")
    (out_dir / "meta.json").write_text(json.dumps({
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "runs": list(results),
        "rows_per_run": {run: int(len(df)) for run, df in results.items()},
        **source_meta,
    }, indent=2) + "\n")

    print(f"wrote {out_dir}/{{fit_vs_val.png,nc_recall_table.tex,meta.json}}")
    print(nc_table[["canonical_recall", "noncanonical_recall", "delta"]].round(3).to_string())


if __name__ == "__main__":
    main()
