import json

import pandas as pd

from soamp.analysis.deck import build_spec, write_pdf, write_spec
from soamp.analysis.facts import analyze, load_thresholds
from soamp.analysis.figures import FIGURES, render_all
from soamp.analysis.narrative import validate_narrative
from tests.analysis.test_facts import FLAT, _run


def _facts(**kw):
    runs = [_run("exp_a", val_mcc=[.30, .33, .35, .36, .36], fit_mcc=[.5, .6, .7, .8, .85], noise=0.02),
            _run("exp_b", val_mcc=FLAT, noise=0.02, n_params=500.0)]
    f, t = analyze(runs, baseline="exp_b", thresholds=load_thresholds(), **kw)
    f["experiments"], f["input_hash"] = ["exp"], "abc12345"
    return f, t


def test_facts_only_deck_has_the_deterministic_slides():
    f, _ = _facts()
    titles = [s.title for s in build_spec(f, None)]
    assert titles[1] == "Ranking" and "Fit vs validation" in titles and "Training curves" in titles
    assert any(t.startswith("Versus baseline") for t in titles) and "Things to be suspicious of" in titles
    assert "What failed" not in titles


def test_failed_cells_and_cost_force_their_slides():
    f, _ = _facts(failed_cells=[{"cell": "cell9", "state": "lost", "detail": "rc=10"}])
    f["cost"] = {"estimate_usd": 0.24, "train_hours": 0.06}
    titles = [s.title for s in build_spec(f, None)]
    assert "What failed" in titles and "Cost" in titles


def test_narrative_slides_and_next_experiments_are_included_in_order():
    f, _ = _facts()
    raw = {"title": "Result", "summary": ["exp_a is ahead of exp_b on validation MCC"],
           "slides": [{"title": "Reading", "bullets": ["gap is large"], "figure": "fit_vs_val"}],
           "next_experiments": [{"title": "Try small", "rationale": "cheap check",
                                 "request": {"experiment": "next_one", "representations": {"peptide": ["rdkit"], "organism": ["vocab"]},
                                             "architectures": [{"name": "baseline_classifier", "epochs": 5}]}}]}
    n, problems = validate_narrative(raw, f, FIGURES)
    assert problems == []
    titles = [s.title for s in build_spec(f, n)]
    assert titles[0] == "Result" and titles[2] == "Reading" and titles[-1].startswith("Next experiment 1")


def test_figures_spec_and_pdf_are_written(tmp_path):
    f, t = _facts()
    render_all(f, t, tmp_path / "figures")
    assert {p.name for p in (tmp_path / "figures").iterdir()} == {"ranking.png", "fit_vs_val.png", "curves.png"}
    slides = build_spec(f, None)
    spec = write_spec(slides, tmp_path / "deck_spec.json")
    assert len(json.loads(spec.read_text())) == len(slides)
    pdf = write_pdf(slides, tmp_path / "figures", tmp_path / "deck.pdf")
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF") and data.count(b"/Type /Page\n") + data.count(b"/Type /Page ") >= len(slides)


def test_pdf_without_any_figures_still_builds(tmp_path):
    pdf = write_pdf(build_spec({"experiments": ["e"], "n_runs": 0, "runs": [], "ranking": [], "flags": [], "global_flags": [],
                                "failed_cells": [{"cell": "c", "state": "lost"}]}, None), tmp_path, tmp_path / "d.pdf")
    assert pdf.stat().st_size > 1000
