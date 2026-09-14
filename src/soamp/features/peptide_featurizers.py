"""Pluggable peptide featurization strategies, selected by a plain method
name via a Protocol + concrete-classes + one build_*() dispatch function
pattern (a dedicated error type raised on an unknown method).

Both concrete strategies below produce a fixed-length, *named* float vector
per peptide_id -- identical output shape to the pre-existing
build_peptide_feature_rows(), so they flow unchanged through the generic
soamp.features.scaling.fit_scaler/apply_scaler and
soamp.features.peptide.index_feature_rows_by_peptide_id pipeline regardless
of which method produced them.
"""
from typing import Protocol

from soamp.features.peptide import DESCRIPTOR_NAMES, build_peptide_feature_rows


class PeptideFeaturizerError(ValueError):
    """Raised when an unknown peptide featurization method is requested."""


class PeptideFeaturizer(Protocol):
    feature_names: list[str]

    def fit(self, fit_unique_peptides: list[dict]) -> None: ...

    def transform(self, unique_peptides: list[dict]) -> list[dict]:
        """[{"peptide_id": ..., <feature_names[0]>: v0, ...}, ...] --
        one row per unique_peptides entry."""
        ...


class RDKitDescriptorFeaturizer:
    """Thin adapter over soamp.features.peptide's existing pure functions --
    no new RDKit logic here."""

    def __init__(self, descriptor_names: list[str] = DESCRIPTOR_NAMES) -> None:
        self.feature_names = list(descriptor_names)

    def fit(self, fit_unique_peptides: list[dict]) -> None:
        pass  # deterministic function of SMILES, no learned params

    def transform(self, unique_peptides: list[dict]) -> list[dict]:
        return build_peptide_feature_rows(unique_peptides, self.feature_names)


class PeptideCLMFeaturizer:
    """Frozen aaronfeller/PeptideCLM-23M-all embeddings, mean-pooled over
    non-padding tokens -- lifted from
    scripts/EDA/02_peptideclm_clustering.ipynb::embed_smiles_batch.

    Tokenizer/model load lazily on first transform() (not __init__), so
    constructing an instance for config validation or dispatch doesn't force
    a checkpoint download -- which also means `device` is only resolved once
    the model is actually needed.

    `device` goes through soamp.utils.device.resolve_device ("auto" picks CUDA
    when available). This forward pass is the one genuinely GPU-bound step in
    the pipeline -- the downstream classifier is a small MLP -- so on a CUDA
    runtime it is the difference between minutes and seconds for a full corpus.

    `cache`: pass a shared dict to reuse embeddings across multiple
    PeptideCLMFeaturizer instances/calls -- e.g. k-fold cross-validation,
    where soamp.data.factory.build_dataset(row_groups=...) constructs a
    fresh instance every fold, but the same peptide_id recurs across folds'
    fit/val partitions. Embeddings are frozen/deterministic, so caching by
    peptide_id is exact, not an approximation: a cache hit returns the
    identical vector a fresh compute would have produced. Defaults to a
    private, empty, per-instance dict, so a caller that never passes
    `cache=` (e.g. pipeline/features/01_build_peptide_features.py's single
    call) sees no behavior change.
    """

    HIDDEN_SIZE = 768

    def __init__(
        self,
        batch_size: int = 16,
        max_length: int = 512,
        checkpoint: str = "aaronfeller/PeptideCLM-23M-all",
        device: str = "auto",
        cache: dict[str, dict] | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.max_length = max_length
        self.checkpoint = checkpoint
        self.device = device
        self._resolved_device = None
        self.feature_names = [f"dim_{i}" for i in range(self.HIDDEN_SIZE)]
        self._tokenizer = None
        self._model = None
        self._cache: dict[str, dict] = cache if cache is not None else {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from pathlib import Path

        from transformers import AutoModel

        from soamp.features.peptideclm.tokenizer import SMILES_SPE_Tokenizer
        from soamp.utils.device import resolve_device

        assets_dir = Path(__file__).resolve().parent / "peptideclm"
        self._tokenizer = SMILES_SPE_Tokenizer(
            str(assets_dir / "new_vocab.txt"), str(assets_dir / "new_splits.txt")
        )
        self._resolved_device = resolve_device(self.device)
        self._model = AutoModel.from_pretrained(self.checkpoint)
        self._model.to(self._resolved_device)
        self._model.eval()

    def fit(self, fit_unique_peptides: list[dict]) -> None:
        pass  # frozen pretrained weights, nothing to fit

    def transform(self, unique_peptides: list[dict]) -> list[dict]:
        """Cache-checking wrapper -- computes only the peptide_ids missing
        from self._cache, then returns every requested row from the cache
        (including ones a previous call already populated)."""
        to_compute = [p for p in unique_peptides if p["peptide_id"] not in self._cache]
        if to_compute:
            for row in self._transform_uncached(to_compute):
                self._cache[row["peptide_id"]] = row
        return [self._cache[p["peptide_id"]] for p in unique_peptides]

    def _transform_uncached(self, unique_peptides: list[dict]) -> list[dict]:
        import numpy as np
        import torch

        self._ensure_loaded()
        smiles_list = [p["smiles"] for p in unique_peptides]
        # Batch by SMILES length to minimize wasted padding compute.
        order = sorted(range(len(smiles_list)), key=lambda i: len(smiles_list[i]))
        pooled_out = np.zeros((len(smiles_list), self.HIDDEN_SIZE), dtype="float32")
        # no_grad scoped to just this forward pass -- a caller in the same
        # process (e.g. a notebook that featurizes then trains a real model)
        # must not have its own training grad computation disabled by a
        # process-wide torch.set_grad_enabled(False) left on after this call.
        with torch.no_grad():
            for start in range(0, len(order), self.batch_size):
                batch_positions = order[start : start + self.batch_size]
                batch_smiles = [smiles_list[i] for i in batch_positions]
                encoded = self._tokenizer(
                    batch_smiles, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_length,
                )
                encoded = {k: v.to(self._resolved_device) for k, v in encoded.items()}
                output = self._model(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled_out[batch_positions] = pooled.cpu().numpy()

        return [
            {"peptide_id": peptide["peptide_id"], **dict(zip(self.feature_names, pooled_out[i].tolist()))}
            for i, peptide in enumerate(unique_peptides)
        ]


def build_peptide_featurizer(method: str, **method_kwargs) -> PeptideFeaturizer:
    if method == "rdkit_descriptors":
        return RDKitDescriptorFeaturizer(**method_kwargs)
    if method == "peptideclm_embedding":
        return PeptideCLMFeaturizer(**method_kwargs)
    raise PeptideFeaturizerError(f"unknown peptide featurization method: {method!r}")
