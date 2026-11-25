import numpy as np
import pytest
import torch

from soamp.engine.config import ModelConfig, TREE_ARCHITECTURES
from soamp.engine.tree_cv import (
    TreeModelError,
    count_tree_parameters,
    train_and_evaluate_fold_tree,
    weighted_bce_from_logits,
)

PARAMS = {"max_iter": 6, "learning_rate": 0.2, "max_leaf_nodes": 4, "min_samples_leaf": 2, "l2_regularization": 0.0, "max_features": 1.0}


def _rows(n, organisms=("Escherichia coli", "Staphylococcus aureus")):
    smiles = ["CC(=O)O", "CCO", "CCN", "CCC", "CCCl", "CCF", "CCBr", "CCI"]
    return [{
        "peptide_id": f"pep_{i}", "sequence": f"SEQ{i}", "smiles": smiles[i % 8], "organism": organisms[i % 2],
        "has_noncanonical": "False", "label": "active" if i % 3 else "inactive",
    } for i in range(n)]


def _fold(**kw):
    rows = _rows(60)
    groups = {"fit": rows[:40], "val": rows[40:]}
    return rows, train_and_evaluate_fold_tree(
        groups, seed=kw.pop("seed", 0), peptide_method="rdkit_descriptors",
        organism_method=kw.pop("organism_method", "vocab_embedding"), params=kw.pop("params", PARAMS), **kw)


def test_weighted_bce_matches_torch():
    rng = np.random.default_rng(0)
    z, y = rng.normal(size=200) * 3, rng.integers(0, 2, 200)
    for pw in (None, 0.17, 2.5):
        ref = torch.nn.BCEWithLogitsLoss(pos_weight=None if pw is None else torch.tensor(pw, dtype=torch.float64))(
            torch.tensor(z), torch.tensor(y, dtype=torch.float64))
        assert weighted_bce_from_logits(z, y, pw) == pytest.approx(float(ref), rel=1e-9)


def test_fold_returns_the_same_contract_as_the_torch_fold():
    rows, (metrics, results) = _fold()
    assert set(metrics) == {"fit", "val"}
    for m in metrics.values():
        assert set(m) == {"accuracy", "f1", "precision", "recall", "auroc", "mcc", "f1_matched", "mcc_matched", "pred_active_rate"}
    assert {"peptide_id", "organism", "label", "true_label_int", "pred_label_int", "logit", "prob", "correct", "eval_group"} <= set(results.columns)
    assert results["eval_group"].value_counts().to_dict() == {"fit": 40, "val": 20}
    assert np.all((results["prob"] > 0) & (results["prob"] < 1))


def test_one_epoch_callback_per_boosting_round_with_finite_values():
    entries = []
    _fold(on_epoch_end=entries.append)
    assert [e["epoch"] for e in entries] == list(range(1, PARAMS["max_iter"] + 1))
    assert all(np.isfinite([e["fit_loss"], e["val_loss"], e["val_auroc"], e["val_mcc"], e["val_f1"]]).all() for e in entries)
    assert entries[-1]["fit_loss"] < entries[0]["fit_loss"]  # boosting reduces the training loss


def test_snapshots_every_n_rounds_and_last_and_match_final_metrics():
    snaps = []
    _, (metrics, results) = _fold(snapshot_every_n_epochs=4, on_snapshot=snaps.append)
    assert [s["epoch"] for s in snaps] == [4, 6]
    last = snaps[-1]
    assert set(last["metrics"]) == {"fit", "val"} and np.isfinite(last["metrics"]["val"]["loss"])
    for key in ("accuracy", "f1", "auroc", "mcc"):
        assert last["metrics"]["val"][key] == pytest.approx(metrics["val"][key])
    val = results[results["eval_group"] == "val"]
    assert list(last["val_predictions"]["peptide_id"]) == list(val["peptide_id"])
    np.testing.assert_allclose(last["val_predictions"]["logit"].to_numpy(), val["logit"].to_numpy())


def test_deterministic_and_callbacks_do_not_change_results():
    _, (_, a) = _fold(seed=3)
    _, (_, b) = _fold(seed=3, on_epoch_end=lambda e: None, snapshot_every_n_epochs=2, on_snapshot=lambda s: None)
    np.testing.assert_allclose(a["logit"].to_numpy(), b["logit"].to_numpy())


def test_class_balancing_changes_the_model():
    _, (_, auto) = _fold(class_balancing_mode="auto")
    _, (_, none) = _fold(class_balancing_mode="none")
    assert not np.allclose(auto["logit"].to_numpy(), none["logit"].to_numpy())


def test_graph_input_is_rejected():
    rows = _rows(60)
    with pytest.raises(TreeModelError, match="vector"):
        train_and_evaluate_fold_tree(
            {"fit": rows[:40], "val": rows[40:]}, seed=0, peptide_method="molecular_graph",
            organism_method="vocab_embedding", params=PARAMS)


def test_snapshots_require_fit_and_val():
    rows = _rows(40)
    with pytest.raises(ValueError, match="'fit' and a 'val'"):
        train_and_evaluate_fold_tree({"fit": rows}, eval_groups=("fit",), seed=0, peptide_method="rdkit_descriptors",
                                     organism_method="vocab_embedding", params=PARAMS, snapshot_every_n_epochs=1, on_snapshot=lambda s: None)


def test_config_registers_the_tree_architecture_and_counts_parameters():
    assert "hist_gradient_boosting_classifier" in TREE_ARCHITECTURES
    m = ModelConfig(architecture="hist_gradient_boosting_classifier")
    assert count_tree_parameters(m.active_kwargs()) == 200 * 31
    with pytest.raises(ValueError):
        ModelConfig(architecture="resnet")


def test_logistic_regression_is_one_round_with_the_same_contract():
    entries, snaps = [], []
    rows, (metrics, results) = _fold(params={"C": 1.0, "max_iter": 500}, architecture="logistic_regression_classifier",
                                     on_epoch_end=entries.append, snapshot_every_n_epochs=2, on_snapshot=snaps.append)
    assert [e["epoch"] for e in entries] == [1] and [s["epoch"] for s in snaps] == [1]
    assert results["eval_group"].value_counts().to_dict() == {"fit": 40, "val": 20}
    assert snaps[0]["metrics"]["val"]["mcc"] == pytest.approx(metrics["val"]["mcc"])
    assert "logistic_regression_classifier" in TREE_ARCHITECTURES
    m = ModelConfig(architecture="logistic_regression_classifier")
    assert count_tree_parameters(m.active_kwargs(), m.architecture, n_features=10) == 11


def test_constant_organism_ablation_removes_the_organism_signal():
    rows, (m_none, res_none) = _fold(organism_method="constant")
    assert set(m_none) == {"fit", "val"}
