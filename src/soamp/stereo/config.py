from pathlib import Path

from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Inputs(_Strict):
    raw_jsonl: Path          # DBAASP records (unusualAminoAcids, bonds, termini)
    processed_parquet: Path  # sequence + SMILES + measurements, one row per measurement
    split_parquet: Path      # peptide_id -> split / fold_id


class Outputs(_Strict):
    dir: Path
    residues: str
    peptides: str
    disagreements: str
    twins: str
    twin_mic: str
    summary: str


class Alignment(_Strict):
    identity_thresholds: list[float]
    gap_open: float
    gap_extend: float
    n_jobs: int


class StereoConfig(_Strict):
    inputs: Inputs
    outputs: Outputs
    n_jobs: int
    alignment: Alignment
