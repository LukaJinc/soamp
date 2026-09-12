"""Baseline MLP: concat(peptide descriptor vector, organism representation)
-> hidden Linear+ReLU layers -> single logit. No sigmoid -- paired with
BCEWithLogitsLoss.
"""
from typing import Literal

import torch
from torch import nn


class OrganismEncoderError(ValueError):
    """Raised when the size arg required by the requested output_kind is
    missing."""


class OrganismEncoder(nn.Module):
    """Wraps either an nn.Embedding (index-based organism representation,
    e.g. the learned vocab-embedding strategy) or an nn.Linear projection
    (vector-based representation, e.g. a future k-mer/taxonomy featurizer)
    behind one forward() call, so BaselineClassifier never branches on
    organism strategy itself -- see soamp.features.organism_featurizers for
    the data-side counterpart of this same output_kind contract."""

    def __init__(
        self,
        organism_embed_dim: int,
        *,
        output_kind: Literal["index", "vector"] = "index",
        organism_vocab_size: int | None = None,
        organism_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.output_kind = output_kind
        if output_kind == "index":
            if organism_vocab_size is None:
                raise OrganismEncoderError(
                    "organism_vocab_size is required when output_kind='index'"
                )
            # No padding_idx=0: the reserved "unknown" organism index still
            # gets a learned (if rarely-trained) embedding rather than a
            # hardcoded zero vector, so an OOV organism at inference gets a
            # plausible representation instead of a fixed neutral point.
            self.encoder: nn.Module = nn.Embedding(organism_vocab_size, organism_embed_dim)
        elif output_kind == "vector":
            if organism_feature_dim is None:
                raise OrganismEncoderError(
                    "organism_feature_dim is required when output_kind='vector'"
                )
            self.encoder = nn.Linear(organism_feature_dim, organism_embed_dim)
        else:
            raise OrganismEncoderError(f"unknown output_kind: {output_kind!r}")

    def forward(self, organism_input: torch.Tensor) -> torch.Tensor:
        return self.encoder(organism_input)


class BaselineClassifier(nn.Module):
    def __init__(
        self,
        peptide_feature_dim: int,
        organism_embed_dim: int = 8,
        hidden_dims: list[int] = [32, 16],
        *,
        organism_output_kind: Literal["index", "vector"] = "index",
        organism_vocab_size: int | None = None,
        organism_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.organism_encoder = OrganismEncoder(
            organism_embed_dim,
            output_kind=organism_output_kind,
            organism_vocab_size=organism_vocab_size,
            organism_feature_dim=organism_feature_dim,
        )

        dims = [peptide_feature_dim + organism_embed_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims, dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, peptide_features: torch.Tensor, organism_input: torch.Tensor) -> torch.Tensor:
        organism_embed = self.organism_encoder(organism_input)
        x = torch.cat([peptide_features, organism_embed], dim=-1)
        return self.mlp(x).squeeze(-1)
