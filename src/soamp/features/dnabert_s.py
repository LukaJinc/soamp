"""Frozen DNABERT-S embeddings of DNA sequences (species-aware: a DNABERT-2
model further trained contrastively so sequences from the same species embed
close together -- the property an organism representation needs, unlike a
plain masked-LM DNA embedding whose cosine similarity is ~0.97+ across all
taxonomic ranks).

Run ONLY from pipeline/features/04_embed_organism_16s_dnabert_s.py, on an
environment with `transformers<5`, `einops` and (for GPU speed) `triton`:
the checkpoint's remote modeling code does not load under transformers 5
(hard `triton` import, `config.pad_token_id`, meta-device init). Training
never imports this module -- it reads the small committed embedding table the
script writes.
"""
import numpy as np


def embed_sequences(
    sequences: list[str],
    *,
    checkpoint: str = "zhihan1996/DNABERT-S",
    revision: str | None = None,
    max_length: int = 512,
    batch_size: int = 8,
    device: str = "auto",
) -> np.ndarray:
    """(len(sequences), hidden) float32, mean-pooled over non-padding tokens
    of the last hidden state. Lazy imports so merely importing this module
    never pulls torch/transformers."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from soamp.utils.device import resolve_device

    resolved = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
    model.to(resolved).eval()

    out = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            encoded = tokenizer(
                sequences[start : start + batch_size], return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            encoded = {k: v.to(resolved) for k, v in encoded.items()}
            hidden = model(**encoded)[0]
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out).astype("float32")
