"""XGBoost variant of the tree path (soamp.engine.tree_cv). Config / error-path tests run
anywhere; the real fits need the xgboost package (and libomp on macOS) and are skipped without it."""
import numpy as np
import pytest
from pydantic import ValidationError

from soamp.engine import tree_cv
from soamp.engine.class_balancing import resolve_pos_weight
from soamp.engine.config import KNOWN_ARCHITECTURES, ModelConfig, TREE_ARCHITECTURES, XGBoostClassifierConfig
from soamp.engine.tree_cv import TreeModelError, count_tree_parameters, train_and_evaluate_fold_tree, weighted_bce_from_logits

try:
    import xgboost  # noqa: F401
    HAS_XGB = True
except Exception:   # not installed, or libomp missing on macOS
    HAS_XGB = False
needs_xgb = pytest.mark.skipif(not HAS_XGB, reason="xgboost is not importable here")

ARCH = "xgboost_classifier"
PARAMS = {"n_estimators": 6, "learning_rate": 0.3, "max_depth": 3, "min_child_weight": 1.0, "reg_lambda": 1.0,
          "reg_alpha": 0.0, "gamma": 0.0, "subsample": 1.0, "colsample_bytree": 1.0}


# ---- config (no xgboost needed) -------------------------------------------------------------

def test_architecture_is_registered_as_a_tree_model():
    assert ARCH in TREE_ARCHITECTURES and ARCH in KNOWN_ARCHITECTURES
    m = ModelConfig(architecture=ARCH)
    assert m.active_kwargs() == XGBoostClassifierConfig().model_dump()
    assert set(m.active_kwargs()) == set(PARAMS)


def test_config_defaults_ranges_and_unknown_keys():
    c = XGBoostClassifierConfig()
    assert (c.n_estimators, c.learning_rate, c.max_depth) == (300, 0.1, 6)
    for bad in ({"n_estimators": 0}, {"learning_rate": 0}, {"max_depth": 0}, {"max_depth": 40}, {"subsample": 0},
                {"subsample": 1.5}, {"colsample_bytree": 0}, {"reg_lambda": -1}, {"max_iter": 10}):
        with pytest.raises(ValidationError):
            XGBoostClassifierConfig(**bad)


def test_parameter_count_is_rounds_times_leaf_bound():
    assert count_tree_parameters(PARAMS, ARCH) == 6 * 2 ** 3
    assert count_tree_parameters(ModelConfig(architecture=ARCH).active_kwargs(), ARCH) == 300 * 64
    # the default (HGB) keeps its own rule
    assert count_tree_parameters({"max_iter": 200, "max_leaf_nodes": 31}) == 200 * 31


def test_unknown_tree_architecture_and_a_failing_worker_have_clear_errors(monkeypatch, tmp_path):
    from tests.engine.test_tree_cv import _rows
    rows = _rows(60)
    groups = {"fit": rows[:40], "val": rows[40:]}
    kw = dict(seed=0, peptide_method="rdkit_descriptors", organism_method="vocab_embedding", params=PARAMS)
    with pytest.raises(TreeModelError, match="unknown tree architecture"):
        train_and_evaluate_fold_tree(groups, architecture="lightgbm_classifier", **kw)
    # a worker whose interpreter has no xgboost -> a clear, actionable error (not a stack trace)
    script = tmp_path / "no_xgboost_worker.py"
    script.write_text("import xgboost_that_does_not_exist\n")
    monkeypatch.setattr(tree_cv, "_XGB_WORKER", script)
    with pytest.raises(TreeModelError, match="xgboost worker failed"):
        train_and_evaluate_fold_tree(groups, architecture=ARCH, **kw)
    script.write_text("import xgboost\nimport xgboost_that_does_not_exist\n") if HAS_XGB else script.write_text("import xgboost\n")
    if not HAS_XGB:
        with pytest.raises(TreeModelError, match="needs the xgboost package"):
            train_and_evaluate_fold_tree(groups, architecture=ARCH, **kw)


# ---- real fits ------------------------------------------------------------------------------

def _fold(**kw):
    from tests.engine.test_tree_cv import _rows
    rows = _rows(60)
    groups = {"fit": rows[:40], "val": rows[40:]}
    return rows, train_and_evaluate_fold_tree(
        groups, seed=kw.pop("seed", 0), peptide_method="rdkit_descriptors", organism_method="vocab_embedding",
        params=kw.pop("params", PARAMS), architecture=ARCH, **kw)


@needs_xgb
def test_fold_returns_the_same_contract_as_the_other_architectures():
    _, (metrics, results) = _fold()
    assert set(metrics) == {"fit", "val"}
    for m in metrics.values():
        assert set(m) == {"accuracy", "f1", "precision", "recall", "auroc", "mcc", "f1_matched", "mcc_matched", "pred_active_rate"}
    assert {"peptide_id", "organism", "label", "true_label_int", "pred_label_int", "logit", "prob", "correct", "eval_group"} <= set(results.columns)
    assert results["eval_group"].value_counts().to_dict() == {"fit": 40, "val": 20}
    assert np.all((results["prob"] > 0) & (results["prob"] < 1))


@needs_xgb
def test_one_epoch_callback_per_round_and_boosting_reduces_the_training_loss():
    entries = []
    _fold(on_epoch_end=entries.append)
    assert [e["epoch"] for e in entries] == list(range(1, PARAMS["n_estimators"] + 1))
    assert all(np.isfinite([e["fit_loss"], e["val_loss"], e["val_auroc"], e["val_mcc"], e["val_f1"]]).all() for e in entries)
    assert entries[-1]["fit_loss"] < entries[0]["fit_loss"]


@needs_xgb
def test_last_round_margin_equals_the_final_logits_and_the_loss_is_the_weighted_bce():
    rows, _ = _fold()
    entries, snaps = [], []
    _, (metrics, results) = _fold(on_epoch_end=entries.append, snapshot_every_n_epochs=4, on_snapshot=snaps.append)
    assert [s["epoch"] for s in snaps] == [4, 6]
    val = results[results["eval_group"] == "val"]
    fit = results[results["eval_group"] == "fit"]
    np.testing.assert_allclose(snaps[-1]["val_predictions"]["logit"].to_numpy(), val["logit"].to_numpy(), rtol=1e-5, atol=1e-5)
    for key in ("accuracy", "f1", "auroc", "mcc"):
        assert snaps[-1]["metrics"]["val"][key] == pytest.approx(metrics["val"][key])
    pw = resolve_pos_weight("auto", None, [1 if r["label"] == "active" else 0 for r in rows[:40]])
    assert entries[-1]["val_loss"] == pytest.approx(
        weighted_bce_from_logits(val["logit"].to_numpy(), val["true_label_int"].to_numpy(), pw), rel=1e-5)
    assert entries[-1]["fit_loss"] == pytest.approx(
        weighted_bce_from_logits(fit["logit"].to_numpy(), fit["true_label_int"].to_numpy(), pw), rel=1e-5)


@needs_xgb
def test_deterministic_per_seed_even_with_row_and_column_sampling_and_callbacks_do_not_change_results():
    sampled = {**PARAMS, "subsample": 0.7, "colsample_bytree": 0.7}
    _, (_, a) = _fold(seed=3, params=sampled)
    _, (_, b) = _fold(seed=3, params=sampled, on_epoch_end=lambda e: None, snapshot_every_n_epochs=2, on_snapshot=lambda s: None)
    _, (_, c) = _fold(seed=4, params=sampled)
    np.testing.assert_allclose(a["logit"].to_numpy(), b["logit"].to_numpy())
    assert not np.allclose(a["logit"].to_numpy(), c["logit"].to_numpy())


@needs_xgb
def test_class_balancing_changes_the_model():
    _, (_, auto) = _fold(class_balancing_mode="auto")
    _, (_, none) = _fold(class_balancing_mode="none")
    assert not np.allclose(auto["logit"].to_numpy(), none["logit"].to_numpy())


@needs_xgb
def test_graph_input_is_rejected():
    from tests.engine.test_tree_cv import _rows
    rows = _rows(60)
    with pytest.raises(TreeModelError, match="vector"):
        train_and_evaluate_fold_tree(
            {"fit": rows[:40], "val": rows[40:]}, seed=0, peptide_method="molecular_graph",
            organism_method="vocab_embedding", params=PARAMS, architecture=ARCH)


# ---- the isolated worker itself ---------------------------------------------------------------

def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8)).astype("float32")
    y = ((x[:, 0] + x[:, 1] * x[:, 2] + 0.5 * rng.normal(size=n)) > 0).astype(int)
    return x, y


@needs_xgb
def test_worker_per_round_margins_are_consistent_with_the_final_model():
    x, y = _data()
    out = tree_cv._run_xgb_worker(PARAMS | {"n_estimators": 12}, 0, x[:300], y[:300], None, {"fit": x[:300], "val": x[300:]})
    assert int(out["n_rounds"]) == 12
    assert out["staged_fit"].shape == (12, 300) and out["staged_val"].shape == (12, 100)
    np.testing.assert_allclose(out["staged_val"][-1], out["final_val"], atol=1e-4)
    assert not np.allclose(out["staged_val"][0], out["staged_val"][-1])          # the model moves between rounds


@needs_xgb
def test_worker_respects_sample_weights_and_the_seed():
    x, y = _data()
    sets = {"val": x[300:]}
    w = np.where(y[:300] == 1, 5.0, 1.0)
    plain = tree_cv._run_xgb_worker(PARAMS, 0, x[:300], y[:300], None, sets)["final_val"]
    weighted = tree_cv._run_xgb_worker(PARAMS, 0, x[:300], y[:300], w, sets)["final_val"]
    assert weighted.mean() > plain.mean()                                         # up-weighted positives push logits up
    sampled = PARAMS | {"subsample": 0.6, "colsample_bytree": 0.6}
    a = tree_cv._run_xgb_worker(sampled, 1, x[:300], y[:300], None, sets)["final_val"]
    b = tree_cv._run_xgb_worker(sampled, 1, x[:300], y[:300], None, sets)["final_val"]
    c = tree_cv._run_xgb_worker(sampled, 2, x[:300], y[:300], None, sets)["final_val"]
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


@needs_xgb
def test_xgboost_learns_a_real_signal_and_agrees_in_kind_with_sklearn_hgb():
    from sklearn.metrics import roc_auc_score
    x, y = _data(1200)
    xs, ys, xv, yv = x[:900], y[:900], x[900:], y[900:]
    out = tree_cv._run_xgb_worker(PARAMS | {"n_estimators": 60, "max_depth": 4}, 0, xs, ys, None, {"val": xv})
    hgb = tree_cv.fit_tree(tree_cv.HGB, {"max_iter": 60, "learning_rate": 0.3, "max_leaf_nodes": 16, "min_samples_leaf": 5,
                                         "l2_regularization": 0.0, "max_features": 1.0}, xs, ys, None, 0, {"val": xv})
    auc_x, auc_h = roc_auc_score(yv, out["final_val"]), roc_auc_score(yv, hgb.logits("val"))
    assert auc_x > 0.8 and abs(auc_x - auc_h) < 0.08


def test_worker_script_is_standalone_and_does_not_import_torch_or_sklearn():
    src = tree_cv._XGB_WORKER.read_text()
    for forbidden in ("import torch", "sklearn", "from soamp", "import soamp"):
        assert forbidden not in src, f"the worker must stay free of {forbidden!r} (OpenMP isolation)"
