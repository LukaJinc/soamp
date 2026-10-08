"""The one place a DataLoader is built for a PeptideOrganismDataset, so the
vector and graph peptide input paths pick the right collate function
without any call site branching on it."""
from torch.utils.data import DataLoader

from soamp.data.torch_dataset import PeptideOrganismDataset


def build_loader(
    dataset: PeptideOrganismDataset,
    batch_size: int,
    *,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=dataset.collate_fn,
    )
