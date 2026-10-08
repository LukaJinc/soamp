"""PyTorch Dataset joining classification rows against precomputed peptide
features + a fitted organism featurizer.

Plain constructor args only -- no load_config() call inside the class, so
it stays importable/testable without any CLI/config involvement.
"""
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from soamp.data.graph_batch import collate_graph_batch
from soamp.features.scaling import apply_scaler

if TYPE_CHECKING:
    from soamp.features.molgraph import MolGraph
    from soamp.features.organism_featurizers import OrganismFeaturizer

LABEL_TO_INT = {"inactive": 0, "active": 1}


class PeptideOrganismDatasetError(ValueError):
    """Raised when descriptor_names/scaler_mean/scaler_scale lengths
    disagree, or when a row's peptide_id has no entry in peptide_features."""


class PeptideOrganismDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        peptide_features: dict[str, dict[str, float]],
        descriptor_names: list[str],
        scaler_mean: list[float],
        scaler_scale: list[float],
        organism_featurizer: "OrganismFeaturizer",
        peptide_graphs: "dict[str, MolGraph] | None" = None,
    ) -> None:
        if not (len(descriptor_names) == len(scaler_mean) == len(scaler_scale)):
            raise PeptideOrganismDatasetError(
                f"length mismatch: descriptor_names={len(descriptor_names)}, "
                f"scaler_mean={len(scaler_mean)}, scaler_scale={len(scaler_scale)}"
            )
        self.rows = rows
        self.peptide_features = peptide_features
        self.descriptor_names = descriptor_names
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.organism_featurizer = organism_featurizer
        # Graph mode: the peptide side of an item is a MolGraph (batched by
        # collate_graph_batch) instead of a scaled descriptor vector; pass
        # empty peptide_features / descriptor_names / scaler in that case.
        self.peptide_graphs = peptide_graphs

    @property
    def collate_fn(self):
        """None (torch's default collate) for vector peptide input."""
        return collate_graph_batch if self.peptide_graphs is not None else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple:
        row = self.rows[idx]
        pid = row["peptide_id"]
        if self.peptide_graphs is not None:
            graph = self.peptide_graphs.get(pid)
            if graph is None:
                raise PeptideOrganismDatasetError(
                    f"peptide_id {pid!r} has no entry in peptide_graphs"
                )
            return graph, *self._organism_and_label(row)
        features = self.peptide_features.get(pid)
        if features is None:
            raise PeptideOrganismDatasetError(
                f"peptide_id {pid!r} has no entry in peptide_features"
            )
        raw_values = [features[name] for name in self.descriptor_names]
        scaled_values = apply_scaler(raw_values, self.scaler_mean, self.scaler_scale)

        return (torch.tensor(scaled_values, dtype=torch.float32), *self._organism_and_label(row))

    def _organism_and_label(self, row: dict) -> tuple[torch.Tensor, torch.Tensor]:
        organism_value = self.organism_featurizer.encode(row["organism"])
        organism_dtype = (
            torch.long if self.organism_featurizer.output_kind == "index" else torch.float32
        )
        label = LABEL_TO_INT[row["label"]]
        return (
            torch.tensor(organism_value, dtype=organism_dtype),
            torch.tensor(label, dtype=torch.float32),
        )
