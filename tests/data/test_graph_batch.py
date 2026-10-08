import numpy as np
import torch

from soamp.data.graph_batch import GraphBatch, batch_graphs, collate_graph_batch
from soamp.data.loaders import build_loader
from soamp.features.molgraph import EDGE_DIM, NODE_DIM, smiles_to_graph


def test_batch_offsets_edge_indices_and_assigns_graph_ids():
    g1, g2 = smiles_to_graph("NCC(=O)O"), smiles_to_graph("CC(=O)O")  # 5 and 4 atoms
    batch = batch_graphs([g1, g2])
    assert batch.num_graphs == 2
    assert batch.node_feats.shape == (9, NODE_DIM)
    assert batch.edge_feats.shape[1] == EDGE_DIM
    assert batch.batch.tolist() == [0] * 5 + [1] * 4
    n_edges_1 = g1.edge_index.shape[1]
    assert (batch.edge_index[:, :n_edges_1] == torch.from_numpy(g1.edge_index)).all()
    assert (batch.edge_index[:, n_edges_1:] == torch.from_numpy(g2.edge_index) + 5).all()


def test_graph_batch_to_moves_all_tensors_and_keeps_num_graphs():
    batch = batch_graphs([smiles_to_graph("CCO")]).to("cpu")
    assert isinstance(batch, GraphBatch) and batch.num_graphs == 1
    assert batch.node_feats.device.type == "cpu"


def test_collate_stacks_organisms_and_labels():
    items = [
        (smiles_to_graph("CCO"), torch.tensor(1), torch.tensor(1.0)),
        (smiles_to_graph("CCN"), torch.tensor(2), torch.tensor(0.0)),
    ]
    graphs, organisms, labels = collate_graph_batch(items)
    assert graphs.num_graphs == 2 and organisms.tolist() == [1, 2] and labels.tolist() == [1.0, 0.0]


def test_build_loader_uses_graph_collate_for_graph_datasets():
    from soamp.data.factory import build_dataset

    rows = [
        {"peptide_id": "1", "smiles": "NCC(=O)O", "organism": "Escherichia coli", "label": "active"},
        {"peptide_id": "2", "smiles": "CC(=O)O", "organism": "Staphylococcus aureus", "label": "inactive"},
    ]
    bundle = build_dataset(row_groups={"fit": rows}, peptide_method="molecular_graph")
    graphs, organisms, labels = next(iter(build_loader(bundle.datasets["fit"], 2)))
    assert isinstance(graphs, GraphBatch) and graphs.num_graphs == 2
    assert labels.shape == (2,)


def test_build_loader_default_collate_for_vector_datasets():
    from soamp.data.factory import build_dataset

    rows = [{"peptide_id": "1", "smiles": "CCO", "organism": "Escherichia coli", "label": "active"}]
    bundle = build_dataset(row_groups={"fit": rows})
    peptide, organism, label = next(iter(build_loader(bundle.datasets["fit"], 1)))
    assert isinstance(peptide, torch.Tensor) and peptide.shape == (1, 13)
