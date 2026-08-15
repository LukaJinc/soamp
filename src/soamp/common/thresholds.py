"""Shared organism-specific MIC-activity threshold lookup.

This is the single place the "active vs inactive" cutoff decision lives.
Both the Stage 1 labeling pipeline (`pipeline/labeling/02_binarize_mic_labels.py`)
and the future Stage 4 benchmarking "threshold decider" consume this module
rather than reimplementing the lookup -- keeping the decision logic in one
place instead of duplicated across `src/soamp/labeling/` and a future
`src/soamp/eval/` (per the project's own architecture notes flagging this
as a "two cooks" risk).

The table itself (`config/thresholds/organism_thresholds.csv`) is a
hand-curated, changelog-versioned artifact -- this module only knows how to
read and query it, not how to generate or edit it (see
`pipeline/labeling/build_threshold_template.py` for that).

Each organism has a dual breakpoint: `active_threshold_uM` (at or below =
active) and `inactive_threshold_uM` (at or above = inactive). A gap between
them is an intentional gray zone -- MIC values that fall strictly inside it
are labeled 'uncertain' rather than forced to one side, matching the
susceptible/intermediate/resistant shape of standard breakpoint tables
(e.g. CLSI/EUCAST). Setting both values equal collapses the gray zone to a
single cutoff.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MatchLevel = Literal["species", "genus", "none"]


class ThresholdTableError(ValueError):
    """Raised when the threshold table CSV is structurally invalid."""


@dataclass(frozen=True)
class ThresholdMatch:
    active_threshold_uM: float
    inactive_threshold_uM: float
    match_level: MatchLevel
    match_key: str


@dataclass(frozen=True)
class ThresholdTable:
    species: dict[str, tuple[float, float]]  # match_key -> (active_uM, inactive_uM)
    genus: dict[str, tuple[float, float]]


def _parse_pair(row: dict, path) -> tuple[float, float] | None:
    """Returns (active, inactive) if both columns are filled, None if both
    are blank. Raises ThresholdTableError on a partial fill (exactly one of
    the two columns set -- ambiguous, not a valid state) or on a
    non-numeric value, or if active > inactive."""
    level, match_key = row["level"].strip(), row["match_key"].strip()
    raw_active = (row.get("active_threshold_uM") or "").strip()
    raw_inactive = (row.get("inactive_threshold_uM") or "").strip()

    if not raw_active and not raw_inactive:
        return None
    if bool(raw_active) != bool(raw_inactive):
        raise ThresholdTableError(
            f"({level}, {match_key!r}) in {path}: active_threshold_uM and "
            f"inactive_threshold_uM must be filled together, got "
            f"active={raw_active!r}, inactive={raw_inactive!r}"
        )

    try:
        active = float(raw_active)
        inactive = float(raw_inactive)
    except ValueError as e:
        raise ThresholdTableError(
            f"({level}, {match_key!r}) in {path}: non-numeric threshold "
            f"active={raw_active!r}, inactive={raw_inactive!r}"
        ) from e

    if active > inactive:
        raise ThresholdTableError(
            f"({level}, {match_key!r}) in {path}: active_threshold_uM "
            f"({active}) must be <= inactive_threshold_uM ({inactive})"
        )

    return active, inactive


def load_threshold_table(path: str | Path) -> ThresholdTable:
    """Load `organism_thresholds.csv` and return only the rows with both
    breakpoints filled in. A row with neither filled is legitimate (the
    user hasn't gotten to that organism yet) and is simply excluded from
    the lookup, not treated as an error.

    Raises `ThresholdTableError` on a duplicate (level, match_key) pair, a
    partial fill (exactly one of the two threshold columns set), a
    non-numeric filled value, or active_threshold_uM > inactive_threshold_uM.
    """
    species: dict[str, tuple[float, float]] = {}
    genus: dict[str, tuple[float, float]] = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        seen: set[tuple[str, str]] = set()
        for row in reader:
            level = row["level"].strip()
            match_key = row["match_key"].strip()
            key = (level, match_key)
            if key in seen:
                raise ThresholdTableError(
                    f"Duplicate ({level}, {match_key!r}) row in {path}"
                )
            seen.add(key)

            pair = _parse_pair(row, path)
            if pair is None:
                continue

            if level == "species":
                species[match_key] = pair
            elif level == "genus":
                genus[match_key] = pair
            else:
                raise ThresholdTableError(
                    f"Unknown level {level!r} for {match_key!r} in {path} "
                    "(expected 'species' or 'genus')"
                )

    return ThresholdTable(species=species, genus=genus)


def lookup_threshold(table: ThresholdTable, organism: str) -> ThresholdMatch | None:
    """Look up the active/inactive MIC breakpoint for a dataset `organism`
    string (e.g. "Staphylococcus aureus" or the genus-only "Pseudomonas sp.").

    Precedence: exact species-level match on the full organism string wins;
    else fall back to a genus-level match on the string's first word; else
    `None` (no threshold available for this organism yet).
    """
    organism = organism.strip()
    if organism in table.species:
        active, inactive = table.species[organism]
        return ThresholdMatch(active, inactive, "species", organism)

    genus = organism.split(" ", 1)[0] if organism else ""
    if genus in table.genus:
        active, inactive = table.genus[genus]
        return ThresholdMatch(active, inactive, "genus", genus)

    return None
