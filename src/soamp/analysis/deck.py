"""Deck content + PDF. Deterministic slides come from the facts only; the agent's narrative
(already validated) adds the interpretation. The same ``deck_spec`` drives the PDF written
here and the Claude Slides artifact the agent creates (downloadable as pptx/PDF), so the two
never disagree. Pages are 16:9, text >= 18 pt, for reading on a phone.
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from soamp.analysis.figures import INK, INK2, SURFACE  # noqa: E402
from soamp.analysis.narrative import Narrative  # noqa: E402


@dataclass
class Slide:
    title: str
    bullets: list[str]
    figure: str | None = None  # file name inside the figures dir


def _short(s: str, n: int = 34) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def build_spec(facts: dict, narrative: Narrative | None) -> list[Slide]:
    runs = {r["name"]: r for r in facts.get("runs", [])}
    ranking = facts.get("ranking", [])
    slides: list[Slide] = []
    title = narrative.title if narrative else "Experiment results: " + ", ".join(facts.get("experiments", []))
    summary = list(narrative.summary) if narrative else [f"{facts.get('n_runs', 0)} runs analysed (no narrative supplied)"]
    slides.append(Slide(title, summary))

    if ranking:
        best = ranking[0]["run"]
        slides.append(Slide("Ranking", [
            f"Best by consensus: {_short(best)} (val MCC {runs[best]['last']['val_mcc']['mean']})",
            f"Baseline: {_short(facts.get('baseline') or '-')}",
            f"{facts['n_runs']} runs, each ranked on val loss, val MCC and val F1 at its own final epoch",
        ], "ranking.png"))
    for s in (narrative.slides if narrative else []):
        slides.append(Slide(s.title, list(s.bullets), f"{s.figure}.png" if s.figure else None))
    if ranking:
        slides.append(Slide("Fit vs validation", ["Large gaps between fit and validation MCC mean overfitting."], "fit_vs_val.png"))
        slides.append(Slide("Training curves", ["Validation loss and MCC per epoch (boosting round for trees)."], "curves.png"))

    cmp_lines = []
    for v in facts.get("vs_baseline", [])[:6]:
        d = v.get("delta_mcc")
        if d:
            cmp_lines.append(f"{_short(v['run'], 28)}: ΔMCC {d['mean']:+.3f}, {d['verdict']} ({d['folds_better']}/{d['n_folds']} folds better)")
    if cmp_lines:
        slides.append(Slide(f"Versus baseline {_short(facts.get('baseline') or '', 30)}", cmp_lines + ["tie = |mean difference| below the fold std"]))

    flags = sorted(facts.get("flags", []), key=lambda f: (f["severity"] != "warn", f["id"]))[:6]
    flag_lines = [f"{_short(f['run'], 22)}: {f['message']}" for f in flags] + [g["message"] for g in facts.get("global_flags", [])[:3]]
    if runs:  # a failure deck has no runs to be suspicious of
        slides.append(Slide("Things to be suspicious of", flag_lines or ["no automatic check fired"]))

    if facts.get("failed_cells"):
        slides.append(Slide("What failed", [f"{_short(c['cell'], 40)}: {c['state']} {c.get('detail', '')}".strip() for c in facts["failed_cells"][:8]]))
    if facts.get("cost"):
        c = facts["cost"]
        slides.append(Slide("Cost", [f"estimated ${c['estimate_usd']:.2f}", f"measured training time {c['train_hours']:.2f} h (sum over cells)"]))
    if narrative and narrative.next_experiments:
        for i, e in enumerate(narrative.next_experiments, 1):
            slides.append(Slide(f"Next experiment {i}: {e.title}", [e.rationale, f"request: {e.request.get('experiment')} (ready to run)"]))
    return slides


def write_spec(slides: list[Slide], path: Path) -> Path:
    path.write_text(json.dumps([asdict(s) for s in slides], indent=2, ensure_ascii=False))
    return path


def write_pdf(slides: list[Slide], figures_dir: Path, path: Path) -> Path:
    with PdfPages(path) as pdf:
        for s in slides:
            fig = plt.figure(figsize=(13.33, 7.5), facecolor=SURFACE)
            fig.text(0.04, 0.92, "\n".join(textwrap.wrap(s.title, 60)), color=INK, fontsize=30, va="top", fontweight="bold")
            img = figures_dir / s.figure if s.figure else None
            has_img = img is not None and img.is_file()
            width = 33 if has_img else 62  # characters per line at 18 / 22 pt on a 13.33 in page
            body = "\n\n".join("\n".join(textwrap.wrap("• " + b, width, subsequent_indent="  ")) for b in s.bullets)
            fig.text(0.04, 0.78, body, color=INK2 if has_img else INK, fontsize=18 if has_img else 22, va="top", linespacing=1.35)
            if has_img:
                ax = fig.add_axes([0.40, 0.06, 0.58, 0.74])
                ax.imshow(mpimg.imread(img))
                ax.axis("off")
            pdf.savefig(fig, facecolor=SURFACE)
            plt.close(fig)
    return path
