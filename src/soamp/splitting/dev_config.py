"""Schema for the dev-split clustering (``config/splitting/dev*.yaml``).

Kept apart from ``SplittingConfig`` on purpose: that one configures the
production split (QMAP wrapper, fingerprints); the dev split uses our own
aligner (``soamp.splitting.alignment``) over residue tokens, so the two have
different knobs. ``extra="forbid"``: a typo'd key raises at load time.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from soamp.splitting.tokens import DEFAULT_NAME_EXCEPTIONS, DResidues


class DevSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organisms: list[str]                  # target strains whose peptides are split
    identity_threshold: float             # pairs at/above this are graph edges
    hard_link_identity: float = 0.9       # pairs at/above this are merged into one node first
    gap_open: float = 5
    gap_extension: float = 1
    d_residues: DResidues = "separate"    # 'separate' | 'map_to_L' (see soamp.splitting.tokens)
    name_exceptions: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_NAME_EXCEPTIONS))
    leiden_seed: int = 42
    leiden_iterations: int = -1           # -1: run until convergence
    split_output: Path                    # peptide_id, cluster_id, cluster_kind, split, fold_id

    @field_validator("identity_threshold", "hard_link_identity")
    @classmethod
    def _is_fraction(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("identity thresholds must be in (0, 1]")
        return v
