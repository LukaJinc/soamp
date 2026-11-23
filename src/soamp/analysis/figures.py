"""Phone-readable figures for the deck (matplotlib, Agg; deterministic given the facts).

Colour follows the dataviz skill's validated reference palette, used in its fixed slot order:
categorical slots 1..8 are assigned to *runs by name order* (an entity keeps its colour no
matter how the ranking changes), text uses ink tokens never series colours, one axis per
panel (no dual axes), thin marks, direct labels on bars, legend whenever >= 2 series.
Sizes are chosen for a phone screen: 10x5.6 in at 150 dpi, >= 14 pt text.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MAX_SERIES = 6  # more lines than this stop being readable on a phone: show the best, say so
FIGURES = {"ranking", "fit_vs_val", "curves"}
SIZE = (10, 5.6)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=13)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _short(name: str, n: int = 26) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


_CELL = re.compile(r"^(?P<exp>.+?)__(?P<arch>[a-z]+)__(?P<rep>.+?)__(?P<hash>[0-9a-f]{6})$")


def _labels(names: list[str]) -> dict[str, str]:
    """Readable, distinguishing labels. Launcher-made run names (<experiment>__<arch>__<reps>__<hash>)
    become "<arch> · <reps>" (the hash is added only if two labels would collide); other names lose
    the prefix shared by all of them (cut at an underscore) so what is left tells runs apart."""
    out: dict[str, str] = {}
    for n in names:
        m = _CELL.match(n)
        if m:
            out[n] = f"{m['arch']} · {m['rep']}"
    dup = {v for v in out.values() if list(out.values()).count(v) > 1}
    for n, v in list(out.items()):
        if v in dup:
            out[n] = f"{v} · {_CELL.match(n)['hash']}"
    rest = [n for n in names if n not in out]
    pre = ""
    if len(rest) > 1:
        first = rest[0].split("_")
        for i in range(len(first)):
            cand = "_".join(first[: i + 1]) + "_"
            if all(n.startswith(cand) and len(n) > len(cand) for n in rest):
                pre = cand
            else:
                break
    out.update({n: n[len(pre):] for n in rest})
    return {n: _short(out[n], 28) for n in names}


def _colors(facts: dict) -> dict[str, str]:
    """Stable entity -> colour: by sorted run name, never by rank."""
    return {n: SERIES[i % len(SERIES)] for i, n in enumerate(sorted(r["name"] for r in facts["runs"]))}


def _ranked(facts: dict) -> list[str]:
    return [r["run"] for r in facts["ranking"]]


def ranking(facts: dict, path: Path) -> Path:
    runs = {r["name"]: r for r in facts["runs"]}
    order = _ranked(facts)[:10]
    fig, ax = plt.subplots(figsize=SIZE, dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = range(len(order))
    means = [runs[n]["last"]["val_mcc"]["mean"] for n in order]
    stds = [runs[n]["last"]["val_mcc"]["std"] or 0 for n in order]
    ax.barh(list(y), means, xerr=stds, color=SERIES[0], height=0.55, error_kw={"ecolor": INK2, "elinewidth": 1.2, "capsize": 3})
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.008, i, f"{m:.3f}", va="center", color=INK, fontsize=14)
    lab = _labels(order)
    ax.set_yticks(list(y), [lab[n] + (" (baseline)" if n == facts.get("baseline") else "") for n in order], color=INK, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("validation MCC, mean ± std over folds (final epoch)", color=INK2, fontsize=14)
    ax.set_title("Ranking by consensus of val loss, MCC and F1", color=INK, fontsize=17, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def fit_vs_val(facts: dict, path: Path) -> Path:
    runs = {r["name"]: r for r in facts["runs"]}
    order = _ranked(facts)[:8]
    fig, ax = plt.subplots(figsize=SIZE, dpi=150, facecolor=SURFACE)
    _style(ax)
    w = 0.38
    xs = range(len(order))
    fit = [runs[n]["last"]["fit_mcc"]["mean"] for n in order]
    val = [runs[n]["last"]["val_mcc"]["mean"] for n in order]
    ax.bar([x - w / 2 - 0.01 for x in xs], fit, w, color=SERIES[0], label="fit (train folds)")
    ax.bar([x + w / 2 + 0.01 for x in xs], val, w, color=SERIES[1], label="validation (held-out fold)")
    for x, f, v in zip(xs, fit, val):
        ax.text(x - w / 2 - 0.01, f + 0.01, f"{f:.2f}", ha="center", color=INK, fontsize=12)
        ax.text(x + w / 2 + 0.01, v + 0.01, f"{v:.2f}", ha="center", color=INK, fontsize=12)
    lab = _labels(order)
    ax.set_xticks(list(xs), [lab[n] for n in order], rotation=25, ha="right", color=INK, fontsize=12)
    ax.set_ylabel("MCC (final epoch)", color=INK2, fontsize=14)
    ax.set_title("Fit vs validation: the gap is overfitting", color=INK, fontsize=17, loc="left", pad=34)
    ax.legend(frameon=False, fontsize=13, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def curves(facts: dict, tidy: pd.DataFrame, path: Path) -> Path:
    """Two small multiples sharing the epoch axis: val loss (lower better) and val MCC (higher better)."""
    col = _colors(facts)
    shown = _ranked(facts)[:MAX_SERIES]
    lab = _labels(sorted(shown))
    fig, axes = plt.subplots(1, 2, figsize=SIZE, dpi=150, facecolor=SURFACE, sharex=True)
    for ax, metric, label in zip(axes, ("loss", "mcc"), ("validation loss", "validation MCC")):
        _style(ax)
        for n in sorted(shown):
            s = tidy[(tidy.run == n) & (tidy.eval_group == "val") & (tidy.metric == metric)].sort_values("epoch")
            if len(s) > 1:
                ax.plot(s["epoch"], s["mean"], color=col[n], linewidth=2, label=lab[n])
        ax.set_title(label, color=INK, fontsize=15, loc="left")
        ax.set_xlabel("epoch / boosting round", color=INK2, fontsize=13)
    axes[0].legend(frameon=False, fontsize=11, labelcolor=INK, loc="best")
    note = "" if len(facts["runs"]) <= MAX_SERIES else f"  (top {MAX_SERIES} of {len(facts['runs'])} runs)"
    fig.suptitle("Training curves" + note, color=INK, fontsize=17, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def render_all(facts: dict, tidy: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "ranking": ranking(facts, out_dir / "ranking.png"),
        "fit_vs_val": fit_vs_val(facts, out_dir / "fit_vs_val.png"),
        "curves": curves(facts, tidy, out_dir / "curves.png"),
    }
