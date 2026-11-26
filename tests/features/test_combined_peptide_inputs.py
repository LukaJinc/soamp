"""Combined peptide inputs: pooled molecular-graph vector, PeptideCLM ++
pooled graph, and the PeptideCLM ++ trainable-GNN hybrid."""
from unittest.mock import patch

import numpy as np
import pytest
import torch

from soamp.data.factory import DatasetFactoryError, build_dataset
from soamp.data.graph_batch import GraphVectorBatch
from soamp.data.loaders import build_loader
from soamp.engine.cross_validation import train_and_evaluate_fold
from soamp.engine.tree_cv import TreeModelError, train_and_evaluate_fold_tree
from soamp.features.molgraph import POOLED_FEATURE_NAMES, pooled_graph_features, smiles_to_graph
from soamp.features.peptide_featurizers import PeptideCLMFeaturizer, build_peptide_featurizer
from soamp.model.factory import build_model
from soamp.model.graph_encoder import GraphVectorPeptideClassifier

SMILES = ["CC(N)C(=O)NCC(=O)O", "NCC(=O)O", "CC(C)CC(N)C(=O)O", "NC(CS)C(=O)O", "NC(Cc1ccccc1)C(=O)O", "OC(=O)CCC(N)C(=O)O"]
ORGS = ["Escherichia coli", "Staphylococcus aureus"]


def _fake_clm(self, peptides):
    # deterministic stand-in for the PeptideCLM forward pass (no checkpoint download)
    return [{"peptide_id": p["peptide_id"], **{n: float(len(p["smiles"]) + i) / 10 for i, n in enumerate(self.feature_names)}}
            for p in peptides]


def _rows():
    rows = []
    for i, smi in enumerate(SMILES):
        for j, org in enumerate(ORGS):
            rows.append({"peptide_id": str(i), "smiles": smi, "organism": org,
                         "label": "active" if (i + j) % 2 else "inactive", "has_noncanonical": "False"})
    return {"fit": rows[:8], "val": rows[8:]}


def test_pooled_graph_features_count_atoms_bonds_and_element_pairs():
    v = dict(zip(POOLED_FEATURE_NAMES, pooled_graph_features(smiles_to_graph("CC(N)C(=O)NCC(=O)O"))))
    assert len(v) == len(POOLED_FEATURE_NAMES)
    assert (v["mg_n_atoms"], v["mg_n_bonds"]) == (10, 9)
    assert (v["mg_pair_C_C"], v["mg_pair_C_N"], v["mg_pair_C_O"]) == (3, 3, 3)
    assert v["mg_atom_count_elem_C"] == 5 and v["mg_atom_frac_elem_C"] == pytest.approx(0.5)
    assert v["mg_bond_count_bond_double"] == 2


def test_concat_featurizer_is_peptideclm_then_pooled_graph():
    f = build_peptide_featurizer("peptideclm_molgraph_pooled")
    assert f.input_kind == "vector"
    assert f.feature_names == [f"dim_{i}" for i in range(768)] + POOLED_FEATURE_NAMES
    with patch.object(PeptideCLMFeaturizer, "_transform_uncached", _fake_clm):
        row = f.transform([{"peptide_id": "a", "smiles": SMILES[0]}])[0]
    assert set(f.feature_names) <= set(row) and row["mg_n_atoms"] == 10


def test_trees_run_on_the_pooled_vectors():
    with patch.object(PeptideCLMFeaturizer, "_transform_uncached", _fake_clm):
        for method in ("molecular_graph_pooled", "peptideclm_molgraph_pooled"):
            metrics, df = train_and_evaluate_fold_tree(
                _rows(), seed=0, peptide_method=method, organism_method="vocab_embedding",
                architecture="hist_gradient_boosting_classifier", params={"max_iter": 3, "min_samples_leaf": 1},
            )
            assert set(metrics) == {"fit", "val"} and len(df) == 12


def test_hybrid_dataset_model_and_one_training_epoch():
    with patch.object(PeptideCLMFeaturizer, "_transform_uncached", _fake_clm):
        bundle = build_dataset(row_groups=_rows(), peptide_method="peptideclm_molgraph")
        feat = bundle.featurization
        assert feat.peptide_input_kind == "graph_vector" and feat.peptide_feature_dim == 768
        batch, organisms, labels = next(iter(build_loader(bundle.datasets["fit"], 4)))
        assert isinstance(batch, GraphVectorBatch) and batch.vectors.shape == (4, 768)
        model = build_model(bundle, architecture="baseline_classifier", graph_encoder_kwargs={"out_dim": 16})
        assert isinstance(model, GraphVectorPeptideClassifier)
        assert model.classifier.mlp[0].in_features == 16 + 768 + 8
        assert model(batch, organisms).shape == (4,)
        with pytest.raises(TreeModelError):
            train_and_evaluate_fold_tree(_rows(), seed=0, peptide_method="peptideclm_molgraph",
                                         organism_method="vocab_embedding", params={"max_iter": 2})
        for arch in ("baseline_classifier", "attention_fusion_classifier"):
            metrics, _ = train_and_evaluate_fold(
                _rows(), epochs=1, seed=0, peptide_method="peptideclm_molgraph",
                organism_method="vocab_embedding", architecture=arch, batch_size=4, device="cpu",
            )
            assert np.isfinite(metrics["val"]["mcc"]) or np.isnan(metrics["val"]["mcc"])


def test_cv_only_stub_is_refused_in_artifact_mode(tmp_path):
    import json
    (tmp_path / "ds.csv").write_text("peptide_id,smiles,organism,label,split\n")
    (tmp_path / "vocab.json").write_text(json.dumps({"method": "vocab_embedding", "vocab": {}, "unknown_index": 0}))
    (tmp_path / "scaler.json").write_text(json.dumps({"method": "peptideclm_molgraph", "cv_only": True}))
    (tmp_path / "val.json").write_text(json.dumps({"val_peptide_ids": []}))
    with pytest.raises(DatasetFactoryError, match="CV-only"):
        build_dataset(data_dir=tmp_path, classification_dataset_filename="ds.csv", organism_vocab_filename="vocab.json",
                      peptide_feature_scaler_filename="scaler.json", val_split_filename="val.json")
