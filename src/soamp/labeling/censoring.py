"""Recover left/right/range censoring bounds lost by the curation pipeline.

`pipeline/curation/06_assay_filter_units.py` collapses every one-sided
bound (`<X`, `>X`) and two-sided range into a single `mic_value_uM` tagged
`mic_type='censored'` in the final dataset, discarding which direction the
bound points. Every such row corresponds to *exactly one* raw DBAASP
`targetActivity` record (any (peptide, organism) pair with 2+ qualifying
raw measurements becomes `mic_type='averaged'` instead -- see 06's grouping
logic), so the original bound is fully recoverable by re-walking
`.cache/dbaasp_raw.jsonl` with the same filters 06 already applies and
re-parsing the raw `concentration` string.

This module only re-derives bounds for rows the caller identifies as
`mic_type='censored'` -- it does not reprocess the whole raw cache or
duplicate 06's SMILES resolution (molecular weights for the rare µg/mL-unit
conversions are passed in, taken directly from the final dataset's already
-computed `molecular_weight` column, since that value is per-peptide and
already resolved there).
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from soamp.curation.parse_dbaasp import DBAASPPeptide
from soamp.curation.taxonomy import classify_species
from soamp.curation.units import classify_censoring, parse_activity, precision, ug_ml_to_uM

ALLOWED_UNITS = ("µM", "µg/ml")

RecoveryStatus = Literal["recovered", "zero_matches", "multi_matches"]


@dataclass(frozen=True)
class CensorBounds:
    status: RecoveryStatus
    censor_type: str  # 'censored' (one-sided) or 'ranged' (two-sided); '' if not recovered
    min_uM: float  # 0.0 / inf sentinels if not recovered (see recover_bounds docstring)
    max_uM: float


def recover_bounds(
    raw_jsonl_path: str | Path,
    censored_pairs: set[tuple[int, str]],
    molecular_weights: dict[int, float | None],
    log=None,
) -> dict[tuple[int, str], CensorBounds]:
    """For each (peptide_id, organism) in `censored_pairs`, find the single
    raw DBAASP targetActivity record that produced its `mic_type='censored'`
    row and recover its true (min, max) bound in uM.

    Returns one entry per pair in `censored_pairs` (never a partial dict):
    pairs where recovery fails (zero or >1 matching raw measurements, which
    should not happen given the schema invariant above, but data can
    surprise you) get the widest-possible bound `(0.0, inf)` with a
    `status` other than "recovered", so downstream label derivation treats
    them as maximally uncertain rather than silently guessing.
    """
    wanted_pids = {pid for pid, _ in censored_pairs}
    candidates: dict[tuple[int, str], list[tuple[str, float, float]]] = {}

    with open(raw_jsonl_path) as f:
        for line in f:
            raw = json.loads(line)
            pid = raw["id"]
            if pid not in wanted_pids:
                continue
            p = DBAASPPeptide(raw)
            for t in p.target_activities:
                group = (t.get("activityMeasureGroup") or {}).get("name")
                if group != "MIC":
                    continue
                species_name = (t.get("targetSpecies") or {}).get("name")
                domain, _taxid, binomial = classify_species(species_name)
                if domain != "Bacteria":
                    continue
                pair = (pid, binomial)
                if pair not in censored_pairs:
                    continue

                unit = (t.get("unit") or {}).get("name")
                if unit not in ALLOWED_UNITS:
                    continue

                min_v, max_v = parse_activity(t.get("concentration"))
                if isinstance(min_v, float) and math.isnan(min_v):
                    continue

                if unit == "µg/ml":
                    mw = molecular_weights.get(pid)
                    if not mw:
                        continue
                    min_v, max_v = ug_ml_to_uM(min_v, max_v, mw)
                    if isinstance(min_v, float) and math.isnan(min_v):
                        continue

                censor = classify_censoring(min_v, max_v, t.get("concentration"))
                if censor == "exact":
                    continue

                candidates.setdefault(pair, []).append((censor, min_v, max_v))

    out: dict[tuple[int, str], CensorBounds] = {}
    n_zero = n_multi = 0
    for pair in censored_pairs:
        matches = candidates.get(pair, [])
        if len(matches) == 1:
            censor, min_v, max_v = matches[0]
            out[pair] = CensorBounds(
                status="recovered",
                censor_type=censor,
                min_uM=precision(min_v),
                max_uM=precision(max_v),
            )
        elif len(matches) == 0:
            n_zero += 1
            out[pair] = CensorBounds("zero_matches", "", 0.0, float("inf"))
        else:
            n_multi += 1
            out[pair] = CensorBounds("multi_matches", "", 0.0, float("inf"))

    if log:
        n_recovered = len(censored_pairs) - n_zero - n_multi
        log.info(
            f"Recovered bounds for {n_recovered}/{len(censored_pairs)} censored pairs "
            f"({n_zero} zero-match, {n_multi} multi-match, treated as maximally uncertain)"
        )

    return out
