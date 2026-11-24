"""Schema for config/data/recompute_dev_v2.yaml (``extra="forbid"``: a typo'd key raises)."""
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from soamp.utils.config import REPO_ROOT


class RecomputePaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processed_parquet: Path = Path("data/dev/dbaasp_dev_processed.parquet")   # read-only input
    out_dir: Path = Path("data/dev_v2")
    reports_dir: Path = Path("reports/dev_v2")


class RecomputeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: RecomputePaths = RecomputePaths()
    organisms: list[str]
    # Regression guard: how many rows of this scope cannot be recomputed (measured: 1 empty
    # concentration + 2 plain numbers without a unit). A different number stops the run so a
    # parsing change cannot silently drop or keep rows.
    expected_dropped_rows: int
    bogus_mw_below: float = 100.0           # implied MW (g/mol) under which DBAASP's weight is unusable
    recomputed_filename: str = "dbaasp_dev_recomputed.parquet"
    dropped_filename: str = "recompute_dropped_rows.csv"
    bogus_mw_filename: str = "dbaasp_bogus_mw_peptides.csv"
    report_filename: str = "mic_recompute_report.md"

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else REPO_ROOT / p
