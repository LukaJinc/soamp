"""Frozen DNABERT-S embeddings of DNA sequences (species-aware: a DNABERT-2
model further trained contrastively so sequences from the same species embed
close together -- the property an organism representation needs, unlike a
plain masked-LM DNA embedding whose cosine similarity is ~0.97+ across all
taxonomic ranks).

Run ONLY from pipeline/features/04_embed_organism_16s_dnabert_s.py, on an
environment with `transformers<5` and `einops`: the checkpoint's remote
modeling code does not load under transformers 5 (hard `triton` import,
`config.pad_token_id`, meta-device init). Training never imports this module
-- it reads the small committed embedding table the script writes.

The remote code's bundled Triton flash-attention kernel targets triton 2.0 and
crashes on current runtimes (`tl.dot() got an unexpected keyword argument
'trans_b'`), so `_disable_triton_attention` switches the model to the
pure-PyTorch attention path the same file already ships (same maths, CPU or
GPU, no triton needed).
"""
import sys

import numpy as np


def _disable_triton_attention(model) -> None:
    """bert_layers.py's BertUnpadSelfAttention computes attention in PyTorch
    when the module-level `flash_attn_qkvpacked_func` is None (it reads that
    global at call time), so patching it routes every layer to the fallback."""
    module = sys.modules[type(model.encoder.layer[0].attention.self).__module__]
    if not hasattr(module, "flash_attn_qkvpacked_func"):
        raise RuntimeError(
            f"{module.__name__} has no `flash_attn_qkvpacked_func`; the DNABERT-S "
            "remote code changed -- check the pinned organism_16s_embedding.revision."
        )
    module.flash_attn_qkvpacked_func = None


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
    _disable_triton_attention(model)
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
