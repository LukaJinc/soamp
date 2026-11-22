"""Generate the deck's "5 · Models" slides from reports/dev_experiments_metrics.csv.

Writes one <section> HTML file per slide (Slides artifact format) into <out-dir>/project/slides/.
Every number on a slide is computed from the CSV (built by build_dev_experiment_facts.py), never
typed. Charts are inline SVG in the deck palette; fit = teal solid, val = amber dashed.

usage: python scripts/slides/build_models_chapter.py --out-dir <folder>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INK, MUTE, GRID, CARD, PAPER2 = "#14213D", "#4A5568", "#D5DAE2", "#FBFAF6", "#E8ECF1"
FIT, VAL = "#0A8A74", "#B4530A"   # validated pair (validate_palette.py, surface #F6F5F1)
FONT = "font-family=\"Inter, Arial, sans-serif\""
SLIDE_IDS = ["part1", "part2", "part3", "labels", "foldbalance", "verdict", "modelsetup", "modelarch", "reps_intro", "pclm_base", "pclm_sweep", "pclm_cw", "rdkit_runs", "graph_runs", "best_table", "best_folds", "fitval", "overfit", "prodvsdev", "classweight", "sweep", "reprs", "modelfindings"]

LABELS = {
    "dev_peptideclm_vocab_cv": "Baseline · 5 epochs",
    "dev_peptideclm_vocab_cv_long": "Baseline · 100 epochs",
    "dev_sweep_wd1e-3": "Weight decay 1e-3",
    "dev_sweep_wd1e-2": "Weight decay 1e-2",
    "dev_sweep_lr1e-4": "Learning rate 1e-4",
    "dev_sweep_lr3e-4_wd1e-3": "LR 3e-4 + decay 1e-3",
    "dev_sweep_small": "Small model",
    "dev_sweep_small_wd1e-3": "Small + decay 1e-3",
    "dev_cb_none_small_wd1e-3": "No class weight, small + decay 1e-3 · 40 epochs",
    "dev_repr_rdkit_vocab_small": "RDKit features, small + decay 1e-3 · 100 epochs",
    "dev_repr_molgraph_vocab_small": "Mol. graph, small + decay 1e-3 · 100 epochs",
}
METRIC_TITLES = {"loss": "Loss", "auroc": "AUROC", "mcc": "MCC", "f1": "F1", "precision": "Precision", "recall": "Recall"}
OVERFIT_PANELS = ["loss", "mcc", "f1", "precision", "recall"]   # decision metrics first; AUROC is in the table


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def val_at(t: pd.DataFrame, run: str, epoch: int | None, group: str, metric: str) -> float:
    g = t[(t.run == run) & (t.eval_group == group) & (t.metric == metric)]
    epoch = g.epoch.max() if epoch is None else epoch
    return float(g[g.epoch == epoch]["mean"].iloc[0])


# ---------------------------------------------------------------- svg helpers
def svg_text(x, y, s, size=24, fill=INK, anchor="start", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" {FONT}>{s}</text>')


def box(x, y, w, h, lines, fill=CARD, stroke=GRID, text_fill=INK, size=24, weight="400"):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2"/>']
    n = len(lines)
    lh = size * 1.25
    y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.35
    for i, line in enumerate(lines):
        out.append(svg_text(x + w / 2, y0 + i * lh, line, size=size, fill=text_fill, anchor="middle",
                            weight=weight if i == 0 else "400"))
    return "".join(out)


def arrow(x1, y1, x2, y2, color=MUTE):
    head = 11
    if y1 == y2:
        d = 1 if x2 > x1 else -1
        tip = f"{x2},{y2} {x2 - d * head * 1.6},{y2 - head} {x2 - d * head * 1.6},{y2 + head}"
    else:
        d = 1 if y2 > y1 else -1
        tip = f"{x2},{y2} {x2 - head},{y2 - d * head * 1.6} {x2 + head},{y2 - d * head * 1.6}"
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2 - (0 if y1 != y2 else (head * 1.4 if x2 > x1 else -head * 1.4))}" '
            f'y2="{y2 - (0 if y1 == y2 else (head * 1.4 if y2 > y1 else -head * 1.4))}" stroke="{color}" stroke-width="3"/>'
            f'<polygon points="{tip}" fill="{color}"/>')


def nice(v):
    return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:.0f}"


def line_panel(x0, y0, w, h, title, series, ylim, yticks, peak=None, shade_from=None, shade_label=None):
    """series: {'fit': (xs, ys), 'val': (xs, ys)}; direct labels at the line ends."""
    pad_l, pad_r, pad_t, pad_b = 66, 64, 40, 34
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    xs_all = sorted({x for s in series.values() for x in s[0]})
    xmin, xmax = xs_all[0], xs_all[-1]
    sx = lambda x: x0 + pad_l + (x - xmin) / (xmax - xmin) * pw
    sy = lambda y: y0 + pad_t + (1 - (y - ylim[0]) / (ylim[1] - ylim[0])) * ph
    out = [svg_text(x0 + pad_l, y0 + 24, title, size=26, weight="600")]
    if shade_from is not None:   # "overfit zone": from the best-val epoch to the end, behind the lines
        out.append(f'<rect x="{sx(shade_from):.1f}" y="{y0 + pad_t}" width="{sx(xmax) - sx(shade_from):.1f}" height="{ph}" fill="{VAL}" fill-opacity="0.13"/>')
        out.append(f'<line x1="{sx(shade_from):.1f}" y1="{y0 + pad_t}" x2="{sx(shade_from):.1f}" y2="{y0 + pad_t + ph}" stroke="{VAL}" stroke-width="2" stroke-dasharray="4 4"/>')
        if shade_label:
            out.append(svg_text(sx(xmax) - 8, y0 + pad_t + 28, shade_label, size=24, fill=VAL, anchor="end", weight="600"))
    for t in yticks:
        out.append(f'<line x1="{sx(xmin)}" y1="{sy(t):.1f}" x2="{sx(xmax)}" y2="{sy(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(svg_text(x0 + pad_l - 10, sy(t) + 8, nice(t), size=24, fill=MUTE, anchor="end"))
    for t in (xmin, (xmin + xmax) // 2 // 5 * 5 if xmax > 40 else xmax, xmax):
        out.append(svg_text(sx(t), y0 + h - 6, str(int(t)), size=24, fill=MUTE, anchor="middle"))
    for name, color, dash in (("fit", FIT, ""), ("val", VAL, "10 6")):
        xs, ys = series[name]
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3.5" '
                   f'stroke-linejoin="round" stroke-linecap="round"{f" stroke-dasharray={chr(34)}{dash}{chr(34)}" if dash else ""}/>')
        out.append(svg_text(sx(xs[-1]) + 10, sy(ys[-1]) + 8, name, size=24, fill=INK, weight="600"))
    if peak is not None:
        px, py = peak
        out.append(f'<circle cx="{sx(px):.1f}" cy="{sy(py):.1f}" r="7" fill="{VAL}" stroke="#F6F5F1" stroke-width="2"/>')
    return "".join(out)


def dumbbell_panel(x0, y0, w, rows, title, xlim, xticks, label_w=300, show_labels=True):
    """rows: [(label, fit, val)]; one row per cell, fit dot (teal) joined to val dot (amber)."""
    pad_t, pad_b, pad_r = 44, 34, 20
    h_row = 58
    plot_x0, plot_x1 = x0 + label_w, x0 + w - pad_r
    sx = lambda v: plot_x0 + (v - xlim[0]) / (xlim[1] - xlim[0]) * (plot_x1 - plot_x0)
    out = [svg_text(plot_x0, y0 + 24, title, size=26, weight="600")]
    top = y0 + pad_t
    for t in xticks:
        out.append(f'<line x1="{sx(t):.1f}" y1="{top - 6}" x2="{sx(t):.1f}" y2="{top + h_row * len(rows)}" stroke="{GRID}" stroke-width="1"/>')
        out.append(svg_text(sx(t), top + h_row * len(rows) + 26, nice(t), size=24, fill=MUTE, anchor="middle"))
    for i, (label, fit, val) in enumerate(rows):
        cy = top + h_row * i + h_row / 2
        if show_labels:
            out.append(svg_text(x0 + label_w - 16, cy + 8, label, size=24, anchor="end"))
        out.append(f'<line x1="{sx(fit):.1f}" y1="{cy}" x2="{sx(val):.1f}" y2="{cy}" stroke="{MUTE}" stroke-width="3"/>')
        out.append(f'<circle cx="{sx(fit):.1f}" cy="{cy}" r="10" fill="{FIT}" stroke="#F6F5F1" stroke-width="2"/>')
        out.append(f'<circle cx="{sx(val):.1f}" cy="{cy}" r="10" fill="{VAL}" stroke="#F6F5F1" stroke-width="2"/>')
    return "".join(out)


def svg_wrap(w, h, label, inner):
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="{label}">{inner}</svg>'


def legend(x, y):
    return (f'<line x1="{x}" y1="{y}" x2="{x + 56}" y2="{y}" stroke="{FIT}" stroke-width="3.5"/>' + svg_text(x + 68, y + 8, "fit (4 training folds)", 24)
            + f'<line x1="{x + 360}" y1="{y}" x2="{x + 416}" y2="{y}" stroke="{VAL}" stroke-width="3.5" stroke-dasharray="10 6"/>'
            + svg_text(x + 428, y + 8, "val (held-out fold)", 24))


def nice_ylim(vals, step=0.1):
    lo, hi = np.floor(min(vals) / step) * step, np.ceil(max(vals) / step) * step
    return (round(lo, 2), round(hi, 2)), [round(lo, 2), round((lo + hi) / 2, 2), round(hi, 2)]


def curve_panels(t, run, shade=True, notes=()):
    """Five fit/val curve panels (loss, MCC, F1, precision, recall) of one run + legend and notes in the 6th cell."""
    g = t[t.run == run]
    series = {m: {grp: (lambda s: (s.epoch.tolist(), s["mean"].tolist()))(
        g[(g.eval_group == grp) & (g.metric == m)].sort_values("epoch")) for grp in ("fit", "val")} for m in OVERFIT_PANELS}
    vloss = series["loss"]["val"]
    e_best_loss = vloss[0][int(np.argmin(vloss[1]))]
    pos = {"loss": (0, 0), "mcc": (562, 0), "f1": (1124, 0), "precision": (0, 310), "recall": (562, 310)}
    panels = []
    for m, (x0, y0) in pos.items():
        allv = series[m]["fit"][1] + series[m]["val"][1]
        ylim, ticks = nice_ylim(allv, 0.5 if m == "loss" else 0.1)
        vs = series[m]["val"]
        best = min(zip(*vs), key=lambda p: p[1]) if m == "loss" else max(zip(*vs), key=lambda p: p[1])
        panels.append(line_panel(x0, y0, 540, 280, METRIC_TITLES[m], series[m], ylim, ticks, peak=best,
                                 shade_from=e_best_loss if shade else None,
                                 shade_label="overfit zone" if shade and m == "loss" else None))
    panels.append(legend_stacked(1130, 372))
    y = 462
    if shade:
        for ln in wrap_lines(f"Shaded: after the best val loss (epoch {int(e_best_loss)})."):
            panels.append(svg_text(1124, y, ln, 24, MUTE)); y += 32
    for ln in wrap_lines("Dot: best val epoch of each metric."):
        panels.append(svg_text(1124, y, ln, 24, MUTE)); y += 32
    y += 6
    for n in notes:
        for ln in wrap_lines(n):
            panels.append(svg_text(1124, y, ln, 24, INK, weight="600")); y += 32
    return "".join(panels)


def legend_stacked(x, y):
    """Two-row fit/val legend for narrow cells (the one-row `legend` is ~900 px wide)."""
    return (f'<line x1="{x}" y1="{y}" x2="{x + 56}" y2="{y}" stroke="{FIT}" stroke-width="3.5"/>' + svg_text(x + 68, y + 8, "fit (4 training folds)", 24)
            + f'<line x1="{x}" y1="{y + 38}" x2="{x + 56}" y2="{y + 38}" stroke="{VAL}" stroke-width="3.5" stroke-dasharray="10 6"/>'
            + svg_text(x + 68, y + 46, "val (held-out fold)", 24))


def wrap_lines(text, width=34):
    import textwrap
    return textwrap.wrap(text, width=width)


FOLD_SERIES = [("precision", "Precision", "#2A5DB0", "circle"), ("recall", "Recall", "#C46A00", "square"), ("mcc", "MCC", "#7B3FA0", "triangle")]


def marker(kind, cx, cy, color, opacity=1.0, r=8):
    if kind == "circle":
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" fill-opacity="{opacity}"/>'
    if kind == "square":
        return f'<rect x="{cx - r:.1f}" y="{cy - r:.1f}" width="{2 * r}" height="{2 * r}" fill="{color}" fill-opacity="{opacity}"/>'
    return f'<polygon points="{cx:.1f},{cy - r - 1:.1f} {cx - r - 1:.1f},{cy + r:.1f} {cx + r + 1:.1f},{cy + r:.1f}" fill="{color}" fill-opacity="{opacity}"/>'


def fold_panel(x0, y0, w, h, title, f):
    """Notebook-style (06_cv_featurization_analysis, section 5): x = CV fold, y in [0, 1]; precision / recall / MCC;
    fit solid, val dashed and faded. `f` has columns val_fold_id, eval_group, precision, recall, mcc."""
    pad_l, pad_r, pad_t, pad_b = 66, 20, 44, 40
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    fold_ids = sorted(f.val_fold_id.unique())
    sx = lambda i: x0 + pad_l + 20 + i / (len(fold_ids) - 1) * (pw - 40)
    sy = lambda v: y0 + pad_t + (1 - v) * ph
    out = [svg_text(x0 + pad_l, y0 + 24, title, size=26, weight="600")]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{x0 + pad_l}" y1="{sy(tick):.1f}" x2="{x0 + w - pad_r}" y2="{sy(tick):.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(svg_text(x0 + pad_l - 10, sy(tick) + 8, nice(tick), size=24, fill=MUTE, anchor="end"))
    for i, fid in enumerate(fold_ids):
        out.append(svg_text(sx(i), y0 + h - 8, str(int(fid)), size=24, fill=MUTE, anchor="middle"))
    for col, _, color, kind in FOLD_SERIES:
        for grp, dash, op in (("fit", "", 1.0), ("val", "10 6", 0.6)):
            g = f[f.eval_group == grp].sort_values("val_fold_id")
            pts = [(sx(i), sy(float(v))) for i, v in enumerate(g[col])]
            out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{color}" stroke-opacity="{op}" '
                       f'stroke-width="3.5" stroke-linejoin="round"{f" stroke-dasharray={chr(34)}{dash}{chr(34)}" if dash else ""}/>')
            out.extend(marker(kind, x, y, color, op) for x, y in pts)
    return "".join(out)


def fold_legend(x, y):
    out, cx = [], x
    for _, label, color, kind in FOLD_SERIES:
        out.append(marker(kind, cx + 10, y, color) + svg_text(cx + 30, y + 8, label, 24))
        cx += 30 + 14 * len(label) + 40
    cx += 20
    out.append(f'<line x1="{cx}" y1="{y}" x2="{cx + 56}" y2="{y}" stroke="{INK}" stroke-width="3.5"/>' + svg_text(cx + 68, y + 8, "fit (4 training folds)", 24))
    cx += 68 + 14 * 22 + 40
    out.append(f'<line x1="{cx}" y1="{y}" x2="{cx + 56}" y2="{y}" stroke="{INK}" stroke-opacity="0.6" stroke-width="3.5" stroke-dasharray="10 6"/>' + svg_text(cx + 68, y + 8, "val (held-out fold)", 24))
    return "".join(out)


def runs_strip(items):
    """One line naming the wandb runs / config behind a slide."""
    return '<p class="note">' + " &nbsp;·&nbsp; ".join(items) + "</p>"


# ---------------------------------------------------------------- slides
PARTS = ["Processing", "Clustering and split", "Experiments"]


def tracker(on: int) -> str:
    """Three chips, the current part highlighted; same markup on every slide of the talk."""
    chips = "".join(f'<span class="on">{i + 1} {name}</span>' if i + 1 == on else f"<span>{i + 1} {name}</span>"
                    for i, name in enumerate(PARTS))
    return f'<div class="tracker">{chips}</div>'


def slide(sid, cls, h2, body, notes, part=3):
    cls_attr = f' class="{cls}"' if cls else ""
    return (f'<section id="{sid}"{cls_attr}>\n{tracker(part)}\n<h2>{h2}</h2>\n{body}\n'
            f'<aside>{notes}</aside>\n</section>\n')


def divider(sid, part, title, lead, notes):
    return (f'<section id="{sid}" class="dark" style="justify-content:center">\n{tracker(part)}\n'
            f'<p class="partnum">{part}</p>\n<h1>{title}</h1>\n<p class="lead">{lead}</p>\n<aside>{notes}</aside>\n</section>\n')


def build(t: pd.DataFrame, facts: dict) -> dict[str, str]:
    runs = list(dict.fromkeys(t.run))
    params = {r["run"]: r["n_parameters"] for r in facts["runs"]}
    big, small = params["dev_peptideclm_vocab_cv"], params["dev_sweep_small"]
    ds = pd.read_csv(REPO_ROOT / "data/dev/mic_classification_dataset.csv")
    n_pairs, n_pep = len(ds), ds.peptide_id.nunique()
    n_train, n_test = int((ds.split == "train").sum()), int((ds.split == "test").sum())
    act = (ds.label == "active").mean() * 100
    out = {}

    # ---- part dividers, labelling (part 1), fold balance (part 2)
    import ast
    log = {}
    for line in (REPO_ROOT / "reports/build_dev_classification_dataset_log.txt").read_text().splitlines():
        if ": " in line and not line.startswith(("===", "Wrote")):
            k, v = line.split(": ", 1)
            try:
                log[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                log[k] = v
    proc = pd.read_parquet(REPO_ROOT / "data/dev/dbaasp_dev_processed.parquet", columns=["target_species"])
    n_records = int(proc.target_species.isin(["Escherichia coli ATCC 25922", "Staphylococcus aureus ATCC 25923",
                                              "Pseudomonas aeruginosa ATCC 27853"]).sum())
    split_df = pd.read_csv(REPO_ROOT / "data/dev/peptide_split.csv")
    ds["bucket"] = np.where(ds.split == "test", "test", "fold_" + ds.peptide_id.map(split_df.set_index("peptide_id").fold_id).fillna(-1).astype(int).astype(str))

    out["part1"] = divider("part1", 1, "Data processing", f"From 25,892 raw DBAASP records to {n_pairs:,} labelled peptide–strain pairs",
                           "Part 1: what we did with the raw dump (SMILES, activity rows, unusual residues) and how activity becomes an active/inactive label.")
    out["part2"] = divider("part2", 2, "Clustering and split", "Whole clusters go to one test set and five folds, so related peptides never sit on both sides of a boundary",
                           "Part 2: alignment identity graph for linear peptides, Murcko ring systems for nonlinear ones, one joint split, and the leakage checks.")
    out["part3"] = divider("part3", 3, "Experiments", f"{len(runs)} cross-validation runs on this data: are we overfitting or underfitting?",
                           "Part 3: the model, fit versus val for every metric, the training curves, why production looked different, and what regularisation changes.")

    rl, cd = log["record_labels"], log["censor_directions"]
    n_cens = cd.get("gt", 0) + cd.get("lt", 0)
    out["labels"] = slide(
        "labels", "alt", f"From {n_records:,} activity records to {log['labelled_pairs']:,} labelled pairs",
        f'''<div class="row">
<div class="col" style="flex:1.1">
<table style="font-size:26px">
<tr><th style="width:64%">Step</th><th>Count</th></tr>
<tr><td>Activity records of the 3 strains</td><td>{n_records:,}</td></tr>
<tr><td>MIC measure only</td><td>{log['records']:,}</td></tr>
<tr><td>Peptide–strain pairs</td><td>{log['peptide_organism_pairs']:,}</td></tr>
<tr><td><b>Labelled active or inactive</b></td><td><b>{log['labelled_pairs']:,}</b></td></tr>
<tr><td>Dropped: uncertain or conflicting</td><td>{log['peptide_organism_pairs'] - log['labelled_pairs']:,}</td></tr>
</table>
<p class="note">Labels: {log['labels']['active']:,} active · {log['labels']['inactive']:,} inactive ({log['labels']['active'] / log['labelled_pairs'] * 100:.1f}% active), from {log['peptides']:,} peptides.</p>
</div>
<div class="col" style="flex:1">
<div class="card"><h3>Cutoffs</h3><p>Active: below 25 µg/ml. Inactive: above 100 µg/ml (DBAASP's tools). In between: dropped.</p></div>
<div class="card"><h3>Units</h3><p>DBAASP <b>activity</b> is in µg/ml, so the cutoffs apply directly.</p></div>
<div class="card"><h3>Censored values</h3><p>{n_cens:,} records read <b>&gt;x</b> or <b>&lt;x</b> in <b>concentration</b> and stay one-sided.</p></div>
</div>
</div>''',
        f"Record level: {rl.get('active', 0):,} active, {rl.get('inactive', 0):,} inactive, {rl.get('uncertain', 0):,} uncertain. Records of one peptide-strain pair are merged: agreeing labels keep that label, disagreeing ones drop the pair. The 25/100 cutoffs replaced an earlier 32/128 pair that had no literature source (CHANGELOG v0.2.0). The split of Part 2 never used these labels.", part=1)

    rows_fb = []
    for b in [f"fold_{i}" for i in range(5)] + ["test"]:
        g = ds[ds.bucket == b]
        rows_fb.append((b, len(g), len(g) / len(ds) * 100, g.peptide_id.nunique(), (g.label == "active").mean() * 100, g.has_noncanonical.mean() * 100))
    fold_rows = [r for r in rows_fb if r[0] != "test"]
    test_row = rows_fb[-1]
    trs_fb = "".join(f"<tr><td>{b}</td><td>{n:,}</td><td>{sh:.1f}</td><td>{pp:,}</td><td{' class=\"bad\"' if b == 'test' else ''}>{ac:.1f}</td><td>{nl:.1f}</td></tr>" for b, n, sh, pp, ac, nl in rows_fb)
    out["foldbalance"] = slide(
        "foldbalance", "", "On the labelled pairs the sizes hold but the label share does not",
        f'''<table style="font-size:26px"><tr><th style="width:16%">Bucket</th><th>Pairs</th><th>% of pairs</th><th>Peptides</th><th>Active %</th><th>Nonlinear %</th></tr>{trs_fb}</table>
<div class="callout" style="font-size:26px">The split used no labels. Fold sizes stay within {min(r[2] for r in fold_rows):.1f}–{max(r[2] for r in fold_rows):.1f}% of the pairs, but the active share runs {min(r[4] for r in fold_rows):.1f}–{max(r[4] for r in fold_rows):.1f}% across folds and is <b>{test_row[4]:.1f}%</b> in test. Expect fold-to-fold spread in CV, and a test set with more actives.</div>''',
        "Part 2 slides (record-level balance of exactly 16.00% per fold) refer to all activity records. The experiments use only the labelled pairs, so the balance is re-measured here. Peptides can have records in several strains, so pair counts and peptide counts differ.", part=2)

    # ---- verdict (first slide of part 3)
    lg, small_wd = "dev_peptideclm_vocab_cv_long", "dev_sweep_small_wd1e-3"
    fm = [val_at(t, r, None, "fit", "mcc") for r in runs]
    vm_all = [val_at(t, r, None, "val", "mcc") for r in runs]
    gaps_all = [a - b for a, b in zip(fm, vm_all)]
    lg_val_loss = t[(t.run == lg) & (t.eval_group == "val") & (t.metric == "loss")].set_index("epoch")["mean"]
    e_min = int(lg_val_loss.idxmin())
    out["verdict"] = slide(
        "verdict", "alt", "Verdict: overfit from about epoch 5, not underfit",
        f'''<div class="row">
<div class="card" style="flex:1.3"><p class="note">MCC, baseline after 100 epochs</p>
<div class="pair"><div><p class="hero fitc">{val_at(t, lg, None, 'fit', 'mcc'):.2f}</p><p class="lab">fit</p></div><div><p class="hero valc">{val_at(t, lg, None, 'val', 'mcc'):.2f}</p><p class="lab">val</p></div></div></div>
<div class="card" style="flex:1"><p class="note">Val loss at epoch 100</p>
<p class="hero valc">{lg_val_loss.loc[100]:.2f}</p><p class="lab">was {lg_val_loss.loc[e_min]:.2f} at epoch {e_min}; fit loss falls to {val_at(t, lg, 100, 'fit', 'loss'):.2f}</p></div>
<div class="card" style="flex:1"><p class="note">Fit minus val MCC, all {len(runs)} runs</p>
<p class="hero valc" style="font-size:76px">{min(gaps_all):.2f}–{max(gaps_all):.2f}</p><p class="lab">never close to zero</p></div>
</div>
<div class="row">
<div class="diag no"><span class="mark">✗</span><span>Underfit? No. Fit MCC is {min(fm):.2f}–{max(fm):.2f} and fit loss falls.</span></div>
<div class="diag yes"><span class="mark">✓</span><span>Overfit? Yes. Val loss rises after epoch {e_min} while fit keeps improving.</span></div>
</div>
<p class="note">Regularisation shrinks the gap but val MCC stays at {min(vm_all):.2f}–{max(vm_all):.2f}: the ceiling is the data and the representation, not the model size.</p>''',
        f"The answer first. Baseline run: fit MCC {val_at(t, lg, None, 'fit', 'mcc'):.3f}, val MCC {val_at(t, lg, None, 'val', 'mcc'):.3f} after 100 epochs; val loss is lowest at epoch {e_min} ({lg_val_loss.loc[e_min]:.3f}) and reaches {lg_val_loss.loc[100]:.3f} at epoch 100. Across all runs the fit-minus-val MCC gap is {min(gaps_all):.2f} to {max(gaps_all):.2f}. The remaining slides show the evidence: the full fit-versus-val table, the curves, why production first looked underfit, and what the sweep changes.")

    # 1 setup
    out["modelsetup"] = slide(
        "modelsetup", "", "Setup: what was trained and how it is measured",
        f'''<div class="row">
<div class="col" style="flex:1">
<ul style="font-size:30px">
<li>{n_pairs:,} labelled peptide–strain pairs from {n_pep:,} peptides, 3 strains ({act:.1f}% active).</li>
<li>Active: MIC below 25 µg/ml. Inactive: above 100 µg/ml (DBAASP). In between: dropped.</li>
<li>5-fold CV on the train split ({n_train:,} pairs); the test split ({n_test:,}) is untouched.</li>
<li><b>fit</b> = the 4 training folds, <b>val</b> = the held-out fold. Numbers are means over the 5 folds.</li>
</ul>
</div>
<div class="col" style="flex:1.1">
<table style="font-size:28px">
<tr><th style="width:24%">Metric</th><th>What it shows</th></tr>
<tr><td><b>MCC</b></td><td>Correlation of prediction and truth, −1 to 1; 0 is chance</td></tr>
<tr><td><b>Precision</b></td><td>Of the predicted active, the share truly active</td></tr>
<tr><td><b>Recall</b></td><td>Of the truly active, the share found</td></tr>
<tr><td><b>AUROC</b></td><td>How well the score ranks active above inactive</td></tr>
<tr><td><b>Loss</b></td><td>BCE with pos_weight = inactive / active (class-weighted)</td></tr>
</table>
</div>
</div>''',
        "Precision, recall and MCC use a 0.5 threshold on the sigmoid output; AUROC does not. Loss is class-weighted binary cross-entropy, so it is not the plain log-loss. Fit metrics are computed in eval mode on the training folds, the same way as val.")

    # 2 architecture
    arch = []
    arch += [box(0, 30, 240, 100, ["Peptide SMILES"]), arrow(240, 80, 300, 80),
             box(300, 30, 380, 100, ["PeptideCLM-23M, frozen", "768-d, mean-pooled", "standardised"]), arrow(680, 80, 740, 80),
             box(740, 30, 330, 100, ["Linear 768 → D", "+ LayerNorm"]), arrow(1070, 80, 1150, 80),
             box(0, 170, 240, 100, ["Strain", "3 + unknown id"]), arrow(240, 220, 300, 220),
             box(300, 170, 380, 100, ["Embedding 4 × D"]), arrow(680, 220, 740, 220),
             box(740, 170, 330, 100, ["LayerNorm"]), arrow(1070, 220, 1150, 220),
             box(1150, 30, 420, 240, ["Self-attention over 2 tokens", "+ slot embedding (peptide / strain)", "4 heads, 1 layer", "residual + LayerNorm"],
                 fill=INK, stroke=INK, text_fill="#F6F5F1"),
             arrow(1360, 270, 1360, 320),
             box(1150, 320, 420, 60, ["Concatenate tokens: 2D"]), arrow(1360, 380, 1360, 430),
             box(1150, 430, 420, 60, ["MLP 64 → 32 (ReLU) → 1 logit"]), arrow(1360, 490, 1360, 540),
             box(1150, 540, 420, 60, ["BCE with pos_weight"], fill=PAPER2)]
    arch += [svg_text(0, 350, "D = 128 (baseline): " + f"{big:,} trainable parameters", 28, weight="600"),
             svg_text(0, 392, "D = 32 (small model, MLP 32 → 16): " + f"{small:,} parameters", 28, weight="600"),
             svg_text(0, 456, "PeptideCLM is frozen: its 768-d embedding is computed once per peptide.", 26, MUTE),
             svg_text(0, 494, "Self-attention needs at least two tokens, so it runs on the two", 26, MUTE),
             svg_text(0, 530, "projected tokens and the result is concatenated afterwards.", 26, MUTE),
             svg_text(0, 594, "Peptide token and strain token have the same width D.", 26, MUTE)]
    out["modelarch"] = slide(
        "modelarch", "alt", "Model: two tokens, one self-attention layer, a small MLP head",
        svg_wrap(1664, 620, "Architecture: peptide SMILES through frozen PeptideCLM and a linear layer, strain id through an embedding, both tokens into one self-attention layer, then concatenation, an MLP and a weighted BCE loss", "".join(arch)),
        "AttentionFusionClassifier in src/soamp/model/attention_fusion.py. The strain is one of 3 ids plus an unknown id, so the strain embedding is a lookup table: a new strain would map to the unknown id. D is the projection width: 128 for the baseline, 32 for the small-model sweep cells.")

    # 3 fit vs val table
    metrics = ["mcc", "f1", "precision", "recall", "loss"]
    head = "<tr><th style=\"width:30%\">Run</th>" + "".join(
        f"<th>{METRIC_TITLES[m]} fit</th><th>{METRIC_TITLES[m]} val</th>" for m in metrics) + "<th>MCC gap</th></tr>"
    trs = []
    for r in runs:
        ep = int(t[t.run == r].epoch.max())
        cells = "".join(f"<td>{val_at(t, r, None, 'fit', m):.3f}</td><td style=\"font-weight:600\">{val_at(t, r, None, 'val', m):.3f}</td>" for m in metrics)
        gap = val_at(t, r, None, "fit", "mcc") - val_at(t, r, None, "val", "mcc")
        cells += f'<td style="background:rgba(180,83,10,{min(gap / 0.7, 1) * 0.55:.2f});font-weight:600">{gap:.2f}</td>'
        name = __import__("re").sub(r" · \d+ epochs", "", LABELS[r]) + f" · {ep} ep"
        trs.append(f"<tr><td>{name}</td>{cells}</tr>")
    spread = t[(t.metric == "auroc") & (t.eval_group == "val")].groupby("run").apply(lambda g: g[g.epoch == g.epoch.max()]["std"].iloc[0])
    out["fitval"] = slide(
        "fitval", "alt", "Fit vs val: every metric, every run",
        f'<table style="font-size:24px;line-height:1.1">{head}{"".join(trs)}</table>\n',
        f"Last epoch of each run, mean of 5 folds; amber column = fit minus val MCC; the no-class-weight loss is plain BCE. Val AUROC std across folds is {spread.min():.3f}–{spread.max():.3f}; AUROC columns are in the best-version table. Source: wandb, project soamp, groups dev_featurization_cv_v1, dev_featurization_cv_v3 and dev_sweep_v1; the table is reports/dev_experiments_metrics.csv. The 5-epoch run stored no eval-mode loss, so its loss is recomputed from the saved logits with the trainer's class-weighted loss; the same recomputation reproduces the stored loss of a sweep run to 5 decimals.")

    # 4 overfit curves, 100-epoch run
    long = "dev_peptideclm_vocab_cv_long"
    g = t[t.run == long]
    vl = g[(g.eval_group == "val") & (g.metric == "loss")].sort_values("epoch")
    e_best_loss = vl.epoch.iloc[int(np.argmin(vl["mean"].to_numpy()))]
    panels = curve_panels(t, long)
    e_first = int(g.epoch.min())
    out["overfit"] = slide(
        "overfit", "", "100 epochs: fit keeps improving, val peaks early, then the overfit zone begins",
        svg_wrap(1664, 620, "Fit and val curves over 100 epochs for loss, MCC, F1, precision and recall", panels),
        f"Baseline run dev_peptideclm_vocab_cv_long, snapshots every 5 epochs, mean over 5 folds, eval mode for both fit and val. Val MCC at epoch {e_first}: {val_at(t, long, e_first, 'val', 'mcc'):.3f}; at epoch 100: {val_at(t, long, 100, 'val', 'mcc'):.3f}. Val loss rises from {val_at(t, long, e_first, 'val', 'loss'):.3f} to {val_at(t, long, 100, 'val', 'loss'):.3f} while fit loss falls to {val_at(t, long, 100, 'fit', 'loss'):.3f}. In the per-epoch curves in wandb the val loss minimum is at epoch 3 and the val AUROC peak at epoch 7.")

    # 5 production vs dev
    pv = pd.read_csv(REPO_ROOT / "reports/prod_vs_dev_metrics.csv")
    def pvv(ds, ep, col):
        return float(pv[(pv.dataset == ds) & (pv.epoch == ep)][col].iloc[0])
    cols = [("prod", 5), ("prod", 100), ("dev", 5), ("dev", 100)]
    spec = [("Loss (class-weighted)", "val_loss", "{:.3f}"), ("MCC at 0.5", "mcc", "{:.3f}"), ("F1 at 0.5", "f1", "{:.3f}"),
            ("<b>F1 at a prior-matched threshold</b>", "f1_matched", "<b>{:.3f}</b>"), ("Precision at 0.5", "precision", "{:.3f}"),
            ("Recall at 0.5", "recall", "{:.3f}"), ("Predicted active at 0.5", "pred_active_rate", "{:.3f}")]
    trs = "".join("<tr><td>" + lab + "</td>" + "".join(f"<td>{fmt.format(pvv(d, e, col))}</td>" for d, e in cols) + "</tr>" for lab, col, fmt in spec)
    pw = {d: (pvv(d, 5, "train_rows"), pvv(d, 5, "active_share"), pvv(d, 5, "pos_weight"), pvv(d, 5, "n_fit_per_fold")) for d in ("prod", "dev")}
    out["prodvsdev"] = slide(
        "prodvsdev", "alt", "Production vs dev: F1 rose from calibration",
        f'''<table style="font-size:24px"><tr><th style="width:36%">Val, mean over 5 folds</th><th>Prod · epoch 5</th><th>Prod · epoch 100</th><th>Dev · epoch 5</th><th>Dev · epoch 100</th></tr>{trs}</table>
<div class="callout" style="font-size:26px">In production F1 at 0.5 rose from {pvv('prod', 5, 'f1'):.3f} to {pvv('prod', 100, 'f1'):.3f}, but at a prior-matched threshold it went {pvv('prod', 5, 'f1_matched'):.3f} to {pvv('prod', 100, 'f1_matched'):.3f}. MCC and val loss got worse. The gain is calibration, not learning.</div>''',
        f"Same model (PeptideCLM x vocab, 100-epoch CV). Production: {int(pw['prod'][3]):,} fit rows per fold, {pw['prod'][1] * 100:.0f}% active, class weight {pw['prod'][2]:.3f}. Dev: {int(pw['dev'][3]):,} rows, {pw['dev'][1] * 100:.0f}% active, weight {pw['dev'][2]:.3f}. Production's class weight (inactive / active) is about 0.17, which pushes the model toward predicting inactive early on: it predicts active for 70% of val rows at epoch 5 while 84% are active, so recall is low and precision high. Training moves the predicted-active rate toward the true rate, so recall and F1 at 0.5 rise even though val loss rises and MCC and AUROC fall. A threshold chosen so the predicted-active rate equals the true rate removes that effect. Dev has less drift because its class weight is 0.51. This is one run per dataset; the class-weight explanation is tested on the next slide with a dev run that has no class weighting.")

    # 5b class weight check (dev, small model, auto weight vs none)
    cw_cols = [("dev_small_auto", 4), ("dev_small_auto", 40), ("dev_small_none", 4), ("dev_small_none", 40)]
    cw_spec = [("MCC at 0.5", "mcc"), ("F1 at 0.5", "f1"), ("<b>F1 at a prior-matched threshold</b>", "f1_matched"),
               ("Precision at 0.5", "precision"), ("Recall at 0.5", "recall"), ("Predicted active at 0.5", "pred_active_rate")]
    cw_trs = "".join("<tr><td>" + lab + "</td>" + "".join(f"<td>{pvv(d, e, col):.3f}</td>" for d, e in cw_cols) + "</tr>" for lab, col in cw_spec)
    sa, sn = "dev_sweep_small_wd1e-3", "dev_cb_none_small_wd1e-3"
    out["classweight"] = slide(
        "classweight", "", "Class weight shifts F1 at 0.5, not the ranking",
        f'''<table style="font-size:24px"><tr><th style="width:34%">Val, mean over 5 folds</th><th>Auto weight · epoch 4</th><th>Auto weight · epoch 40</th><th>No weight · epoch 4</th><th>No weight · epoch 40</th></tr>{cw_trs}</table>
<p class="note">PeptideCLM × vocab, small model, decay 1e-3. Auto weight = {pvv('dev_small_auto', 4, 'pos_weight'):.3f}. True active rate in val: {pvv('dev', 5, 'true_active_rate'):.3f}.</p>
<div class="callout" style="font-size:26px">Without the weight the model predicts active for {pvv('dev_small_none', 40, 'pred_active_rate') * 100:.0f}% of val rows (true {pvv('dev', 5, 'true_active_rate') * 100:.0f}%), against {pvv('dev_small_auto', 40, 'pred_active_rate') * 100:.0f}% with it; F1 at 0.5 is {pvv('dev_small_none', 40, 'f1'):.3f} against {pvv('dev_small_auto', 40, 'f1'):.3f}. At the matched threshold: {pvv('dev_small_none', 40, 'f1_matched'):.3f} and {pvv('dev_small_auto', 40, 'f1_matched'):.3f}. MCC is the same.</div>''',
        f"Val loss is not comparable between the two (weighted vs plain BCE). Fit MCC {val_at(t, sn, 40, 'fit', 'mcc'):.2f}, val {val_at(t, sn, 40, 'val', 'mcc'):.2f} without weight: the overfit gap is the same. This tests the explanation of the production-vs-dev slide. The weight only shifts the decision threshold: the ranking (matched-threshold F1 and MCC, AUROC) is unchanged, so F1 at 0.5 depends on the weight but MCC and the matched F1 do not. Class balancing is therefore not the cause of the overfitting; the fit-minus-val gap is the same with and without it. One run per setting, one seed.")

    # 6 sweep dumbbells at epoch 40, decision metrics
    cells = [r for r in runs if r.startswith("dev_sweep_")]
    entries = [("Baseline", long, 40)] + [(LABELS[r], r, 40) for r in cells]
    entries.sort(key=lambda e: -val_at(t, e[1], e[2], "val", "mcc"))
    def rows_for(metric):
        return [(lab, val_at(t, r, ep, "fit", metric), val_at(t, r, ep, "val", metric)) for lab, r, ep in entries]
    sw = (dumbbell_panel(0, 0, 760, rows_for("loss"), "Loss", (0.0, 0.9), [0, 0.3, 0.6, 0.9], label_w=330)
          + dumbbell_panel(760, 0, 460, rows_for("mcc"), "MCC", (0.2, 1.0), [0.2, 0.6, 1.0], label_w=0, show_labels=False)
          + dumbbell_panel(1220, 0, 444, rows_for("f1"), "F1", (0.7, 1.0), [0.7, 0.85, 1.0], label_w=0, show_labels=False)
          + legend(330, 560))
    vm = [val_at(t, r, ep, "val", "mcc") for _, r, ep in entries]
    vf = [val_at(t, r, ep, "val", "f1") for _, r, ep in entries]
    out["sweep"] = slide(
        "sweep", "alt", "Sweep at epoch 40: regularisation shrinks the gap, val MCC and F1 barely move",
        svg_wrap(1664, 600, "Dumbbell charts of fit and val loss, MCC and F1 at epoch 40 for the baseline and six regularisation cells, sorted by val MCC", sw),
        f"Rows are sorted by val MCC, best first. All cells are PeptideCLM x vocab embedding on the dev data, 40 epochs, 5-fold CV; the baseline row is the 100-epoch run's epoch-40 snapshot (constant learning rate, so identical to a 40-epoch run). Val MCC spans {min(vm):.3f}-{max(vm):.3f} and val F1 {min(vf):.3f}-{max(vf):.3f} across the seven rows. Weight decay is Adam's L2 term.")

    # 6b representations at matched epochs (decision metrics)
    rep = [("PeptideCLM · " + f"{params['dev_sweep_small_wd1e-3'] / 1000:.1f}k", "dev_sweep_small_wd1e-3"),
           ("RDKit · " + f"{params['dev_repr_rdkit_vocab_small'] / 1000:.1f}k", "dev_repr_rdkit_vocab_small"),
           ("Mol. graph · " + f"{params['dev_repr_molgraph_vocab_small'] / 1000:.1f}k", "dev_repr_molgraph_vocab_small")]
    def rep_rows(metric, ep=40):
        return [(lab, val_at(t, r, ep, "fit", metric), val_at(t, r, ep, "val", metric)) for lab, r in rep]
    rs = (dumbbell_panel(0, 0, 760, rep_rows("loss"), "Loss", (0.0, 0.9), [0, 0.3, 0.6, 0.9], label_w=330)
          + dumbbell_panel(760, 0, 460, rep_rows("mcc"), "MCC", (0.2, 1.0), [0.2, 0.6, 1.0], label_w=0, show_labels=False)
          + dumbbell_panel(1220, 0, 444, rep_rows("f1"), "F1", (0.7, 1.0), [0.7, 0.85, 1.0], label_w=0, show_labels=False)
          + legend(330, 330))
    r_rdk, r_gnn = "dev_repr_rdkit_vocab_small", "dev_repr_molgraph_vocab_small"
    spread = [float(t[(t.run == r) & (t.eval_group == "val") & (t.metric == "mcc") & (t.epoch == 40)]["std"].iloc[0]) for _, r in rep]
    vmcc40 = [val_at(t, r, 40, "val", "mcc") for _, r in rep]
    out["reprs"] = slide(
        "reprs", "", "Representations: RDKit and the graph do not overfit, PeptideCLM does, val MCC is about equal",
        svg_wrap(1664, 380, "Fit and val loss, MCC and F1 at epoch 40 for PeptideCLM, RDKit descriptors and the molecular graph, all with the small model", rs)
        + f'''<p class="note">Same small model, weight decay 1e-3, vocab strain embedding, epoch 40 (PeptideCLM has no longer small-model run; the others are shown to epoch 100 below). Trainable parameters after the dot.</p>
<div class="callout" style="font-size:28px">Val MCC {min(vmcc40):.2f}–{max(vmcc40):.2f} for the three, inside the fold spread (std {min(spread):.3f}–{max(spread):.3f}). To epoch 100 RDKit and the graph stay flat: val loss {val_at(t, r_rdk, 100, 'val', 'loss'):.2f} and {val_at(t, r_gnn, 100, 'val', 'loss'):.2f}, val MCC {val_at(t, r_rdk, 100, 'val', 'mcc'):.2f} and {val_at(t, r_gnn, 100, 'val', 'mcc'):.2f}, fit MCC only {val_at(t, r_rdk, 100, 'fit', 'mcc'):.2f} and {val_at(t, r_gnn, 100, 'fit', 'mcc'):.2f}.</div>''',
        "The 768-d PeptideCLM embedding adds little validation signal over 13 RDKit descriptors, but its larger input makes the model memorise: fit MCC 0.79 against val 0.35 at epoch 40, while RDKit and the graph have gaps of about 0.13-0.15 and a flat val loss. The two low-capacity representations are not clearly underfit either: their val metrics do not improve with more fit. Best val values over all epochs (picked on the same folds, so optimistic): graph MCC 0.370 and F1 0.774 around epoch 64, loss 0.412 at epoch 70; PeptideCLM loss 0.424 at epoch 2, MCC 0.353; RDKit MCC 0.346. The ceiling looks like the data and the labels, not the representation.")

    # ======== per-representation slides ========
    r5, rl100 = "dev_peptideclm_vocab_cv", "dev_peptideclm_vocab_cv_long"
    r_pc40 = "dev_sweep_small_wd1e-3"
    def card(title, run, extra):
        return (f'<div class="card" style="flex:1"><h3>{title}</h3><p class="big" style="font-size:56px">{params[run] / 1000:.1f}k</p>'
                f'<p class="lab">trainable parameters</p><p style="font-size:26px">{extra}</p></div>')
    n_pc_runs = sum(1 for r in runs if r.startswith(("dev_peptideclm", "dev_sweep", "dev_cb")))
    out["reps_intro"] = slide(
        "reps_intro", "alt", f"Three peptide representations, {len(runs)} runs",
        f'''<div class="row">
{card("PeptideCLM embedding", "dev_peptideclm_vocab_cv", f"Frozen transformer, 768-d. {n_pc_runs} runs: baseline (5 and 100 epochs), a 6-cell regularisation sweep, a class-weight check. Small model: {params['dev_sweep_small'] / 1000:.1f}k.")}
{card("RDKit descriptors", "dev_repr_rdkit_vocab_small", "13 physico-chemical descriptors of the SMILES, standardised. 1 run, small model, 100 epochs.")}
{card("Molecular graph", "dev_repr_molgraph_vocab_small", "Atoms and bonds, a GINE graph network trained end to end. 1 run, small model, 100 epochs.")}
</div>
<p class="note">Strain side is fixed to a learned embedding of the 3 strain ids in every run. All runs: 5-fold CV on the train split, fit = 4 training folds, val = held-out fold, mean over folds. Next slides: one group per representation.</p>''',
        "Parameter counts come from the wandb summary n_parameters of each run. The PeptideCLM baseline and sweep cells use the attention-fusion model with D = 128 (184,577 parameters) except the two 'small' cells (D = 32). RDKit and graph runs use the small model so that capacity is comparable.")

    out["pclm_base"] = slide(
        "pclm_base", "", "PeptideCLM baseline: val peaks in the first epochs, then fit pulls away",
        svg_wrap(1664, 620, "Fit and val curves over 100 epochs of the PeptideCLM baseline", curve_panels(
            t, rl100, shade=True, notes=[f"5 epochs: fit MCC {val_at(t, r5, None, 'fit', 'mcc'):.2f}, val {val_at(t, r5, None, 'val', 'mcc'):.2f}",
                                         f"100 epochs: fit MCC {val_at(t, rl100, None, 'fit', 'mcc'):.2f}, val {val_at(t, rl100, None, 'val', 'mcc'):.2f}"]))
        + runs_strip([f"{LABELS[r5]}: <b>{r5}</b>", f"{LABELS[rl100]}: <b>{rl100}</b>", "config cv_dev_peptideclm_vocab[_long].yaml"]),
        f"PeptideCLM x vocab embedding, D = 128 ({big:,} parameters), 5 folds. The 100-epoch run stores eval-mode snapshots every 5 epochs. The 5-epoch run is a separate run; its fit/val are the end-of-training values. Val loss {val_at(t, rl100, 5, 'val', 'loss'):.3f} at epoch 5 and {val_at(t, rl100, 100, 'val', 'loss'):.3f} at epoch 100.")

    def retitle(html, sid, title):
        import re
        html = re.sub(r'id="[^"]*"', f'id="{sid}"', html, count=1)
        return re.sub(r"<h2>.*?</h2>", f"<h2>{title}</h2>", html, count=1)
    out["pclm_sweep"] = retitle(out["sweep"], "pclm_sweep", "PeptideCLM sweep at epoch 40: regularisation shrinks the gap, val MCC barely moves")
    out["pclm_cw"] = retitle(out["classweight"], "pclm_cw", "PeptideCLM class weight: only F1 at 0.5 moves")

    def single_run(sid, title, run, label, extra_notes, conclusion):
        fm, vm = val_at(t, run, None, "fit", "mcc"), val_at(t, run, None, "val", "mcc")
        return slide(sid, "", title,
            svg_wrap(1664, 620, f"Fit and val curves over 100 epochs, {label}", curve_panels(
                t, run, shade=False, notes=[f"100 epochs: fit MCC {fm:.2f}, val {vm:.2f}", *extra_notes]))
            + runs_strip([f"<b>{run}</b>", f"{params[run]:,} parameters"]),
            conclusion)
    for sid, run, label, title in (("rdkit_runs", r_rdk, "RDKit descriptors", "RDKit descriptors: fit and val stay close, val loss is flat"),
                                   ("graph_runs", r_gnn, "molecular graph", "Molecular graph: fit and val stay close, val loss is flat")):
        out[sid] = single_run(sid, title, run, label,
            [f"val loss {val_at(t, run, 10, 'val', 'loss'):.2f} (ep 10), {val_at(t, run, 100, 'val', 'loss'):.2f} (ep 100)"],
            f"{label.capitalize()} x vocab embedding, small model (D = 32 equivalent), weight decay 1e-3, 100 epochs, snapshots every 2 epochs, mean over 5 folds. Fit MCC {val_at(t, run, 100, 'fit', 'mcc'):.3f}, val MCC {val_at(t, run, 100, 'val', 'mcc'):.3f}; val loss {val_at(t, run, 100, 'val', 'loss'):.3f}. Unlike PeptideCLM there is no overfit zone: val loss does not rise.")

    # ======== best version per representation ========
    REPS = [("PeptideCLM", "pclm", lambda r: r.startswith(("dev_peptideclm", "dev_sweep"))),
            ("RDKit", "rdkit", lambda r: "rdkit" in r), ("Molecular graph", "graph", lambda r: "molgraph" in r)]
    best = {}
    for name, key, pred in REPS:   # best = highest mean val MCC over all snapshots; the no-class-weight run is excluded (loss not comparable)
        v = t[(t.eval_group == "val") & (t.metric == "mcc") & t.run.map(pred) & (t.run != "dev_cb_none_small_wd1e-3")]
        row = v.loc[v["mean"].idxmax()]
        best[key] = (name, row.run, int(row.epoch))
    folds = pd.read_csv(REPO_ROOT / "reports/dev_best_runs_folds.csv")
    for key, (name, run, ep) in best.items():   # the per-fold file must describe the same run and epoch
        f = folds[folds.representation == name]
        assert set(f.run) == {run} and set(f.epoch) == {ep}, f"rerun build_best_run_folds.py: best of {name} is {run} @ {ep}"
    bm = ["mcc", "f1", "precision", "recall", "auroc", "accuracy", "loss"]
    bhead = '<tr><th style="width:21%">Representation · run · epoch</th><th>Set</th>' + "".join(f"<th>{METRIC_TITLES.get(m, m.capitalize())}</th>" for m in bm) + "</tr>"
    btr = []
    for key, (name, run, ep) in best.items():
        for grp in ("fit", "val"):
            lab = f'<b>{name}</b> · epoch {ep}' if grp == "fit" else ""
            cells = "".join(f'<td{" style=\"font-weight:600\"" if grp == "val" else ""}>{val_at(t, run, ep, grp, m):.3f}</td>' for m in bm)
            btr.append(f"<tr><td>{lab}</td><td>{grp}</td>{cells}</tr>")
    best_mcc_std = {k: float(t[(t.run == r) & (t.epoch == e) & (t.eval_group == "val") & (t.metric == "mcc")]["std"].iloc[0]) for k, (_, r, e) in best.items()}
    out["best_table"] = slide(
        "best_table", "alt", "Best version of each representation, every metric",
        f'''<table style="font-size:24px">{bhead}{"".join(btr)}</table>
<p class="note">Best = highest mean val MCC over all epoch snapshots. Mean over 5 folds.</p>
<div class="callout" style="font-size:26px">Val MCC is {min(val_at(t, r, e, "val", "mcc") for _, r, e in best.values()):.2f}–{max(val_at(t, r, e, "val", "mcc") for _, r, e in best.values()):.2f} for all three, inside the fold spread (std {min(best_mcc_std.values()):.2f}–{max(best_mcc_std.values()):.2f}). Epochs were picked on the same folds, so val is optimistic.</div>''',
        f"Runs: " + "; ".join(f"{n}: {r}" for n, r, _ in best.values()) + ". Selection excludes dev_cb_none_small_wd1e-3 (plain BCE, so its loss is not comparable); it reaches val MCC 0.359 at epoch 14, level with the chosen PeptideCLM cell (0.359 at epoch 10). Loss is class-weighted BCE. The test split has not been used. Per-fold values of these three versions are on the next slides (reports/dev_best_runs_folds.csv).")

    panels, vr = "", {}
    for i, (key, (name, run, ep)) in enumerate(best.items()):
        f = folds[folds.representation == name]
        panels += fold_panel(i * 562, 0, 540, 360, f"{name} · epoch {ep}", f)
        v = f[f.eval_group == "val"]
        vr[name] = {m: (float(v[m].min()), float(v[m].max())) for m in ("mcc", "precision", "recall")}
    rng = lambda n, m: f"{vr[n][m][0]:.2f}–{vr[n][m][1]:.2f}"
    panels += fold_legend(0, 392)
    out["best_folds"] = slide(
        "best_folds", "", "Best versions fold by fold: precision, recall and MCC",
        svg_wrap(1664, 420, "Fit and val precision, recall and MCC for each of the 5 CV folds, for the best PeptideCLM, RDKit and molecular-graph versions", panels)
        + runs_strip([f"<b>{r}</b> · epoch {e}" for _, r, e in best.values()])
        + f'''<p class="note">Val MCC across folds: {" · ".join(f"{n} {rng(n, 'mcc')}" for n in vr)}. x-axis: held-out fold.</p>''',
        "Source: reports/dev_best_runs_folds.csv (script build_best_run_folds.py, wandb cv_snapshots table of each run). One x position per CV fold; solid = fit score (4 training folds), dashed = val score (held-out fold). " + " ".join(f"{n}: val precision {rng(n, 'precision')}, recall {rng(n, 'recall')}, MCC {rng(n, 'mcc')}." for n in vr) + " Folds are whole peptide clusters, so fold-to-fold spread reflects cluster composition and active share (see the fold-balance slide).")

    # 7 findings (decisions on val loss, MCC, F1)
    p5, p100, sm, wd = "dev_peptideclm_vocab_cv", long, "dev_sweep_small_wd1e-3", "dev_sweep_wd1e-2"
    cells_all = [(r, None) for r in runs]
    vmcc = {r: val_at(t, r, None, "val", "mcc") for r in runs}
    fmcc = {r: val_at(t, r, None, "fit", "mcc") for r in runs}
    pvl = {r: val_at(t, r, None, "val", "precision") for r in runs}
    rvl = {r: val_at(t, r, None, "val", "recall") for r in runs}
    sm_run = "dev_sweep_small_wd1e-3"
    out["modelfindings"] = slide(
        "modelfindings", "", f"Findings: what the {len(runs)} runs say",
        f'''<div class="row">
<div class="card" style="flex:1"><p class="bign">{val_at(t, p100, None, 'fit', 'mcc'):.2f} → {vmcc[p100]:.2f}</p><p>Fit vs val MCC after 100 epochs. Training longer widens the gap: from 5 to 100 epochs val loss goes {val_at(t, p5, None, 'val', 'loss'):.2f} → {val_at(t, p100, None, 'val', 'loss'):.2f} and val MCC {vmcc[p5]:.2f} → {vmcc[p100]:.2f}.</p></div>
<div class="card" style="flex:1"><p class="big">{vmcc[sm_run]:.2f}</p><p>Best val MCC of the {len(runs)} runs: small model + decay 1e-3 (val loss {val_at(t, sm_run, None, 'val', 'loss'):.2f}, F1 {val_at(t, sm_run, None, 'val', 'f1'):.2f}). Weight decay 1e-2 has the lowest val loss ({val_at(t, wd, None, 'val', 'loss'):.2f}) but MCC {vmcc[wd]:.2f}.</p></div>
<div class="card" style="flex:1"><p class="big">{min(pvl.values()):.2f}–{max(pvl.values()):.2f}</p><p>Val precision, above val recall ({min(rvl.values()):.2f}–{max(rvl.values()):.2f}): the model misses actives more than it over-calls them.</p></div>
</div>
<div class="callout" style="font-size:28px">Decisions use val loss, MCC and F1. F1 at 0.5 partly measures calibration (production slide), so MCC and loss carry more weight. Caveats: last-epoch numbers; test split not used; 3 strains; 25 / 100 µg/ml labels.</div>''',
        "AUROC is logged for every run but not used for decisions. The earlier production-data CV runs (different dataset and labels) also overfit in loss and ranking: fit AUROC about 0.998 at 100 epochs against val 0.77, with val AUROC peaking near 0.80 around epoch 10. The dev data has about half as many training rows per fold and a lower ceiling. The calibration explanation was confirmed with a dev run without class weighting, and the other peptide representations were compared. Next steps: early stopping or dropout, more strains, and a confirmation of the chosen setting on the untouched test split.")
    return out


def main() -> None:
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    t = load(REPO_ROOT / "reports/dev_experiments_metrics.csv")
    facts = json.loads((REPO_ROOT / "reports/slide_figs/dev_experiments_facts.json").read_text())
    slides = build(t, facts)
    d = Path(args.out_dir) / "project" / "slides"
    d.mkdir(parents=True, exist_ok=True)
    for sid in SLIDE_IDS:
        (d / f"{sid}.html").write_text(slides[sid])
        print(f"{sid}: {len(slides[sid]):,} bytes")


if __name__ == "__main__":
    main()
