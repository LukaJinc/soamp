"""Small trainable GIN-style molecular-graph encoder (GINE: edge features are
added to each message), pure torch -- no PyTorch Geometric dependency.

Deliberately small (hidden 64 x 4 layers ~ 40k parameters): weights are shared
across every atom and bond, so size is set by width/depth, not molecule size.
No pretraining, so it must learn chemistry from this dataset's ~12k peptides --
dropout, LayerNorm and weight decay (config) are the regularisers.
"""
import torch
from torch import nn

from soamp.data.graph_batch import GraphBatch

# Sum pooling carries molecule size (mean/max cannot see how many atoms there
# are, and size -- MolWt / HeavyAtomCount -- is among the strongest signals in
# the descriptor baseline). Peptides average ~150 heavy atoms, so dividing by
# this keeps the pooled sum O(1) per unit of hidden activation.
SUM_POOL_SCALE = 100.0


class GINEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        out_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_projections = nn.ModuleList(
            nn.Linear(edge_dim, hidden_dim) for _ in range(num_layers)
        )
        self.update_mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        )
        self.epsilons = nn.ParameterList(nn.Parameter(torch.zeros(1)) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        # mean, max and (scaled) sum readouts concatenated, then projected
        self.readout = nn.Sequential(nn.Linear(3 * hidden_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, graphs: GraphBatch) -> torch.Tensor:
        h = self.node_embedding(graphs.node_feats)
        source, target = graphs.edge_index
        for edge_proj, update, eps, norm in zip(
            self.edge_projections, self.update_mlps, self.epsilons, self.norms
        ):
            messages = torch.relu(h[source] + edge_proj(graphs.edge_feats))
            aggregated = torch.zeros_like(h).index_add_(0, target, messages)
            h = norm(h + self.dropout(update((1.0 + eps) * h + aggregated)))
        pooled = torch.cat(
            [self._mean_pool(h, graphs), self._max_pool(h, graphs), self._sum_pool(h, graphs) / SUM_POOL_SCALE],
            dim=-1,
        )
        return self.readout(pooled)

    @staticmethod
    def _sum_pool(h: torch.Tensor, graphs: GraphBatch) -> torch.Tensor:
        total = torch.zeros(graphs.num_graphs, h.shape[1], device=h.device, dtype=h.dtype)
        return total.index_add_(0, graphs.batch, h)

    @classmethod
    def _mean_pool(cls, h: torch.Tensor, graphs: GraphBatch) -> torch.Tensor:
        total = cls._sum_pool(h, graphs)
        counts = torch.bincount(graphs.batch, minlength=graphs.num_graphs).clamp(min=1)
        return total / counts.unsqueeze(1).to(h.dtype)

    @staticmethod
    def _max_pool(h: torch.Tensor, graphs: GraphBatch) -> torch.Tensor:
        index = graphs.batch.unsqueeze(1).expand_as(h)
        init = torch.full((graphs.num_graphs, h.shape[1]), float("-inf"), device=h.device, dtype=h.dtype)
        return init.scatter_reduce(0, index, h, reduce="amax", include_self=True)


class GraphPeptideClassifier(nn.Module):
    """GINEncoder -> any existing fusion classifier. The wrapped model sees a
    fixed-length peptide vector (the encoder's `out_dim`), exactly like the
    descriptor/PeptideCLM vectors, so BaselineClassifier and
    AttentionFusionClassifier work unchanged."""

    def __init__(self, peptide_encoder: GINEncoder, classifier: nn.Module) -> None:
        super().__init__()
        self.peptide_encoder = peptide_encoder
        self.classifier = classifier

    def forward(self, graphs: GraphBatch, organism_input: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.peptide_encoder(graphs), organism_input)
