---
name: analyze-experiment
description: Analyze finished soamp experiments from W&B and produce the report and slide deck with recommended next experiments. Use when the user asks for results/analysis/comparison of experiments, or when the watcher prints ANALYZE, RETRY or REPORT_FAILURE.
---

# Analyze an experiment

Scripts do all numbers, charts and the PDF; you write only the interpretation, from `facts.json` alone.

## On demand ("how did X do?", "compare X and Y")
1. `pip install -q -e . --no-deps`
2. `python scripts/exp/analyze.py --experiment <group> [--experiment <group2> ...] [--baseline-run <run name>]`
   (default baseline: `dev_sweep_small_wd1e-3`, the CLAUDE.md dev reference; if absent, the best run of the experiment).
   The last line printed is the output directory `D`.
3. Read `D/facts.json` and write `D/narrative.json`:
   ```json
   {"title": "<=80 chars", "summary": ["<=3 bullets"],
    "slides": [{"title": "...", "bullets": ["<=3"], "figure": "ranking|fit_vs_val|curves|null"}],
    "next_experiments": [{"title": "...", "rationale": "...", "request": { <a full, valid request, see run-experiment> }}]}
   ```
   Hard rules (the validator rejects violations and tells you where):
   - Every number must exist in facts.json (any rounding). Do not compute new numbers; at most 6 slides, 3 bullets each.
   - Select by **val loss, val MCC and val F1** (CLAUDE.md decision rule); AUROC is context only.
   - Respect the facts: if a comparison verdict is `tie`, say it is a tie (and mention the folds_better/n_folds direction if useful); never declare a winner inside fold noise. Mention every `warn` flag that matters (overfitting, rising val loss, calibration drift, failed cells), and the single-seed caveat.
   - Recommendations come from the same folds as the ranking, so say they should be confirmed on the held-out test split before being believed; never touch the test split yourself.
   - Each `next_experiments[].request` must be a valid, NEW-named request (<=3), cheap enough for the caps; prefer the experiment that would resolve the biggest uncertainty in the facts.
4. `python scripts/exp/build_deck.py --dir D`
   - `NARRATIVE REJECTED` -> fix narrative.json as listed and re-run (up to 3 times).
   - `UNCHANGED` -> this exact input was already published; just give the user the existing report (W&B artifact `exp_report-<experiment>` in project `soamp-ops`) and stop.
   - `PUBLISHED <artifact>` -> D now has `deck_spec.json`, `deck.pdf`, `figures/*.png`; the PDF is also in the W&B artifact.
5. Create the deck as a **Claude Slides artifact** for the phone: call the Artifact tool with `action: "quickstart"`, `intent: "slides"`, then
   follow its instructions (publish from the Slides type, `auto_open: "after_first_write"`, fill it). Slide titles/bullets come **verbatim from
   `D/deck_spec.json`**, one slide per entry, in order; for entries with a `figure`, use the image `D/figures/<figure>`. Add nothing, change no number.
6. Send a `PushNotification` with the artifact link and a one-line verdict (e.g. "X vs baseline: tie within fold noise; 2 warnings"). Also give the link in your final message, plus the W&B artifact name of the PDF fallback.

## When the watcher prints
- `ANALYZE <experiment>`: do the steps above for that experiment (one deck per line).
- `RETRY <experiment>`: `python scripts/exp/launch.py --retry <experiment>`; report its output; do nothing else.
- `REPORT_FAILURE <experiment>`: nothing finished. Run `python scripts/exp/build_deck.py --failure <experiment>` (it publishes the failure report and marks it delivered), then give the user the failed cells and the likely cause. Common ones: rc 10 setup failed, 11 GPU unusable, 12 data artifact fetch failed, 124 timeout; look at the run_status artifact in `soamp-ops` for the log tail. Never relaunch beyond what `RETRY` allows.
- `JANITOR destroyed ...`: tell the user which instance was destroyed and why, once.
- `STATUS ...` lines are informational.
