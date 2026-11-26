"""Batching of variable-size molecular graphs into one disjoint-union graph.

No PyTorch Geometric dependency: a batch is four plain tensors. GraphBatch
implements `.to(device)` so soamp.engine.trainer.Trainer -- which only ever
calls `peptide_features.to(device)` -- runs graph batches unchanged.
"""
from dataclasses import dataclass

import numpy as np
import torch

from soamp.features.molgraph import MolGraph


@dataclass
class GraphBatch:
    node_feats: torch.Tensor   # (total_nodes, NODE_DIM) float32
    edge_index: torch.Tensor   # (2, total_edges) int64, indices into node_feats
    edge_feats: torch.Tensor   # (total_edges, EDGE_DIM) float32
    batch: torch.Tensor        # (total_nodes,) int64, graph id of every node
    num_graphs: int

    def to(self, device: "torch.device | str") -> "GraphBatch":
        return GraphBatch(
            node_feats=self.node_feats.to(device),
            edge_index=self.edge_index.to(device),
            edge_feats=self.edge_feats.to(device),
            batch=self.batch.to(device),
            num_graphs=self.num_graphs,
        )


def batch_graphs(graphs: list[MolGraph]) -> GraphBatch:
    """Concatenates graphs, shifting each graph's edge indices by the number
    of nodes before it."""
    node_counts = [g.num_nodes for g in graphs]
    offsets = np.concatenate([[0], np.cumsum(node_counts)[:-1]]).astype(np.int64)
    edge_index = np.concatenate(
        [g.edge_index + offset for g, offset in zip(graphs, offsets)], axis=1
    )
    batch = np.repeat(np.arange(len(graphs), dtype=np.int64), node_counts)
    return GraphBatch(
        node_feats=torch.from_numpy(np.concatenate([g.node_feats for g in graphs])),
        edge_index=torch.from_numpy(edge_index),
        edge_feats=torch.from_numpy(np.concatenate([g.edge_feats for g in graphs])),
        batch=torch.from_numpy(batch),
        num_graphs=len(graphs),
    )


def collate_graph_batch(items: list[tuple]) -> tuple[GraphBatch, torch.Tensor, torch.Tensor]:
    """DataLoader collate for PeptideOrganismDataset's graph mode: items are
    (MolGraph, organism_tensor, label_tensor)."""
    graphs, organisms, labels = zip(*items)
    return batch_graphs(list(graphs)), torch.stack(organisms), torch.stack(labels)


@dataclass
class GraphVectorBatch:
    """A graph batch plus a (num_graphs, D) float vector per graph (the
    `graph_vector` peptide input: GNN-encoded graph ++ frozen embedding)."""

    graphs: GraphBatch
    vectors: torch.Tensor

    def to(self, device: "torch.device | str") -> "GraphVectorBatch":
        return GraphVectorBatch(graphs=self.graphs.to(device), vectors=self.vectors.to(device))


def collate_graph_vector_batch(items: list[tuple]) -> tuple[GraphVectorBatch, torch.Tensor, torch.Tensor]:
    """DataLoader collate for graph_vector mode: items are
    ((MolGraph, vector_tensor), organism_tensor, label_tensor)."""
    peptides, organisms, labels = zip(*items)
    graphs, vectors = zip(*peptides)
    return (
        GraphVectorBatch(graphs=batch_graphs(list(graphs)), vectors=torch.stack(vectors)),
        torch.stack(organisms),
        torch.stack(labels),
    )
