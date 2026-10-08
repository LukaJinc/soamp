import numpy as np
import pytest
import torch

from soamp.data.factory import build_dataset
from soamp.data.graph_batch import batch_graphs
from soamp.data.loaders import build_loader
from soamp.features.molgraph import EDGE_DIM, NODE_DIM, MolGraph, smiles_to_graph
from soamp.model.factory import ModelFactoryError, build_model
from soamp.model.graph_encoder import GINEncoder, GraphPeptideClassifier

SMILES = ["NCC(=O)O", "CC(=O)NCC(=O)O", "O=C1CNC(=O)CNC(=O)CN1", "N[C@@H](C)C(=O)O"]


def _batch():
    return batch_graphs([smiles_to_graph(s) for s in SMILES])


def test_encoder_output_shape_gradients_and_no_nans():
    encoder = GINEncoder(NODE_DIM, EDGE_DIM, hidden_dim=32, num_layers=3, out_dim=16)
    out = encoder(_batch())
    assert out.shape == (4, 16) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_default_encoder_is_small():
    n = sum(p.numel() for p in GINEncoder(NODE_DIM, EDGE_DIM).parameters())
    assert n < 60_000


def test_encoder_is_invariant_to_node_ordering():
    graph = smiles_to_graph("CC(=O)NCC(=O)O")
    perm = np.random.default_rng(0).permutation(graph.num_nodes)
    inverse = np.argsort(perm)
    permuted = MolGraph(
        node_feats=graph.node_feats[perm],
        edge_index=inverse[graph.edge_index],
        edge_feats=graph.edge_feats,
    )
    encoder = GINEncoder(NODE_DIM, EDGE_DIM).eval()
    assert torch.allclose(
        encoder(batch_graphs([graph])), encoder(batch_graphs([permuted])), atol=1e-5
    )


def test_encoder_output_depends_on_molecule_size():
    # mean/max pooling alone cannot tell GGG from GGGGG; sum pooling can.
    encoder = GINEncoder(NODE_DIM, EDGE_DIM).eval()
    short, long = smiles_to_graph("NCC(=O)O"), smiles_to_graph("NCC(=O)NCC(=O)NCC(=O)NCC(=O)O")
    out = encoder(batch_graphs([short, long]))
    assert not torch.allclose(out[0], out[1])


@pytest.mark.parametrize("architecture", ["baseline_classifier", "attention_fusion_classifier"])
def test_build_model_graph_path_works_with_both_architectures(architecture):
    rows = [
        {"peptide_id": str(i), "smiles": s, "organism": "Escherichia coli", "label": "active" if i % 2 else "inactive"}
        for i, s in enumerate(SMILES)
    ]
    bundle = build_dataset(row_groups={"fit": rows}, peptide_method="molecular_graph")
    model = build_model(bundle, architecture=architecture, graph_encoder_kwargs={"hidden_dim": 16, "out_dim": 8})
    assert isinstance(model, GraphPeptideClassifier)
    graphs, organisms, labels = next(iter(build_loader(bundle.datasets["fit"], 4)))
    logits = model(graphs, organisms)
    assert logits.shape == (4,) and torch.isfinite(logits).all()


def test_build_model_graph_requires_dims_without_a_bundle():
    with pytest.raises(ModelFactoryError):
        build_model(peptide_input_kind="graph", organism_vocab_size=3)


def test_graph_model_overfits_a_tiny_batch():
    rows = [
        {"peptide_id": str(i), "smiles": s, "organism": "Escherichia coli", "label": "active" if i % 2 else "inactive"}
        for i, s in enumerate(SMILES)
    ]
    bundle = build_dataset(row_groups={"fit": rows}, peptide_method="molecular_graph")
    torch.manual_seed(0)
    model = build_model(bundle, architecture="attention_fusion_classifier",
                        graph_encoder_kwargs={"hidden_dim": 32, "out_dim": 16, "dropout": 0.0})
    graphs, organisms, labels = next(iter(build_loader(bundle.datasets["fit"], 4)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(300):
        optimizer.zero_grad()
        loss = loss_fn(model(graphs, organisms), labels)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05
