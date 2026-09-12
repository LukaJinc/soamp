"""Attention-fusion classifier: projects the peptide and organism
representations to a shared `projection_dim` (so representations of very
different raw dimensionality/scale -- e.g. 13-dim RDKit descriptors vs.
768-dim PeptideCLM embeddings, or an 8-dim organism embedding vs. a much
larger k-mer vector -- contribute on equal footing), treats the two
projections as a 2-token sequence, self-attends across them (so the two
representations can actually condition on each other, not just a fixed
linear combination), then concatenates the attended tokens into the MLP
head.

Self-attention needs >=2 tokens to do anything: attending over a single
concatenated vector degenerates to a fixed linear transform of that one
token -- this is why attention happens BEFORE concatenation here, not
after.
"""
from typing import Literal

import torch
from torch import nn

from soamp.model.baseline_mlp import OrganismEncoder


class AttentionFusionClassifier(nn.Module):
    def __init__(
        self,
        peptide_feature_dim: int,
        projection_dim: int = 128,
        num_attention_heads: int = 4,
        num_attention_layers: int = 1,
        hidden_dims: list[int] = [64, 32],
        *,
        organism_output_kind: Literal["index", "vector"] = "index",
        organism_vocab_size: int | None = None,
        organism_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.peptide_projection = nn.Sequential(
            nn.Linear(peptide_feature_dim, projection_dim), nn.LayerNorm(projection_dim),
        )
        # Reused as-is from baseline_mlp.py -- already handles both
        # index-based and vector-based organism representations projecting
        # to a target dim; here that target is projection_dim, not
        # organism_embed_dim.
        self.organism_encoder = OrganismEncoder(
            projection_dim, output_kind=organism_output_kind,
            organism_vocab_size=organism_vocab_size, organism_feature_dim=organism_feature_dim,
        )
        self.organism_norm = nn.LayerNorm(projection_dim)

        # Learned per-slot embedding (peptide slot vs. organism slot) added
        # before attention -- both tokens live in the same projection_dim
        # space, so this gives attention an explicit "which token is which"
        # signal on top of whatever it infers from content alone.
        self.slot_embedding = nn.Embedding(2, projection_dim)

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(projection_dim, num_attention_heads, batch_first=True)
            for _ in range(num_attention_layers)
        ])
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(projection_dim) for _ in range(num_attention_layers)
        ])

        dims = [projection_dim * 2, *hidden_dims]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims, dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, peptide_features: torch.Tensor, organism_input: torch.Tensor) -> torch.Tensor:
        peptide_token = self.peptide_projection(peptide_features)
        organism_token = self.organism_norm(self.organism_encoder(organism_input))

        tokens = torch.stack([peptide_token, organism_token], dim=1)      # (B, 2, D)
        tokens = tokens + self.slot_embedding.weight.unsqueeze(0)         # (B, 2, D)

        for attention, norm in zip(self.attention_layers, self.attention_norms):
            attended, _ = attention(tokens, tokens, tokens)
            tokens = norm(tokens + attended)

        fused = tokens.reshape(tokens.shape[0], -1)                        # (B, 2*D) concat
        return self.mlp(fused).squeeze(-1)
