from soamp.analysis.narrative import allowed_numbers, unknown_numbers, validate_narrative

FACTS = {"runs": [{"name": "run_a", "last": {"val_mcc": {"mean": 0.3521, "std": 0.0388}}, "n_parameters": 7681, "last_epoch": 40}],
         "global_flags": [{"message": "fit MCC exceeds val MCC by 0.463 at epoch 40"}], "n_runs": 6}
FIGS = {"ranking", "curves"}
REQ = {"experiment": "next_try_v1", "representations": {"peptide": ["rdkit"], "organism": ["vocab"]},
       "architectures": [{"name": "baseline_classifier", "epochs": 20}]}


def _n(**kw):
    base = {"title": "Sweep results", "summary": ["run_a reaches val MCC 0.35 (std 0.039) after 40 epochs"], "slides": [], "next_experiments": []}
    base.update(kw)
    return base


def test_numbers_in_facts_pass_at_any_rounding():
    n, problems = validate_narrative(_n(), FACTS, FIGS)
    assert n is not None and problems == []
    assert unknown_numbers("0.352 and 0.35 and 7681 and 0.463", allowed_numbers(FACTS)) == []


def test_invented_number_is_rejected_with_its_location():
    n, problems = validate_narrative(_n(summary=["MCC improved to 0.61"]), FACTS, FIGS)
    assert any("summary[0]" in p and "0.61" in p for p in problems)


def test_small_counts_are_allowed_but_big_unknown_integers_are_not():
    assert unknown_numbers("all 5 folds, 3 runs", allowed_numbers(FACTS)) == []
    assert unknown_numbers("trained for 90000 steps", allowed_numbers(FACTS)) == ["90000"]


def test_schema_limits_bullets_and_slides():
    n, problems = validate_narrative(_n(slides=[{"title": "t", "bullets": ["a", "b", "c", "d"]}]), FACTS, FIGS)
    assert n is None and problems and problems[0].startswith("schema")


def test_unknown_figure_rejected():
    _, problems = validate_narrative(_n(slides=[{"title": "t", "bullets": ["x"], "figure": "pie"}]), FACTS, FIGS)
    assert any("figure" in p for p in problems)


def test_next_experiment_must_be_a_valid_new_request():
    ok = _n(next_experiments=[{"title": "Try small MLP", "rationale": "cheap", "request": REQ}])
    n, problems = validate_narrative(ok, FACTS, FIGS)
    assert n is not None and problems == []
    n2, problems2 = validate_narrative(_n(next_experiments=[{"title": "x", "rationale": "y", "request": {**REQ, "experiment": "run_a"}}]), FACTS, FIGS)
    assert any("already exists" in p for p in problems2)
    n3, problems3 = validate_narrative(_n(next_experiments=[{"title": "x", "rationale": "y", "request": {"experiment": "bad"}}]), FACTS, FIGS)
    assert n3 is None and problems3[0].startswith("schema")  # an invalid draft is reported, not raised
