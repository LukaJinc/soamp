"""Baseline MLP: concat(peptide descriptor vector, organism embedding) ->
hidden Linear+ReLU layers -> single logit. No sigmoid -- paired with
BCEWithLogitsLoss.
"""
import torch
from torch import nn


class BaselineClassifier(nn.Module):
    def __init__(
        self,
        peptide_feature_dim: int,
        organism_vocab_size: int,
        organism_embed_dim: int = 8,
        hidden_dims: list[int] = [32, 16],
    ) -> None:
        super().__init__()
        # No padding_idx=0: the reserved "unknown" organism index still
        # gets a learned (if rarely-trained) embedding rather than a
        # hardcoded zero vector, so an OOV organism at inference gets a
        # plausible representation instead of a fixed neutral point.
        self.organism_embedding = nn.Embedding(organism_vocab_size, organism_embed_dim)

        dims = [peptide_feature_dim + organism_embed_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims, dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, peptide_features: torch.Tensor, organism_idx: torch.Tensor) -> torch.Tensor:
        organism_embed = self.organism_embedding(organism_idx)
        x = torch.cat([peptide_features, organism_embed], dim=-1)
        return self.mlp(x).squeeze(-1)
