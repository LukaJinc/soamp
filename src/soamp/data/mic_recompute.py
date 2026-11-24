"""Recompute a MIC in ug/ml from DBAASP's free-text ``concentration`` (pure logic, no I/O).

Why: DBAASP's own ``activity`` column is the concentration converted to ug/ml with
DBAASP's molecular weight, and for some peptides that weight is nonsense (17.03,
1.01, 0.0, 45.09 g/mol were found), so the stored ``activity`` is wrong there. We
recompute from ``concentration`` + ``unit`` and use the peptide's own RDKit weight.

Rules, per row (``mic_ug_ml``):

* ``concentration == "NA"``  -> the DBAASP sentinel **1,111,111**, whatever the unit.
  (DBAASP uses it for "not active up to ..."; after the 25/100 cutoffs it is inactive.)
* unit ug/ml -> the number as written.
* unit uM    -> ``uM * MW / 1000`` with MW = RDKit ``MolWt`` of the peptide SMILES.
* anything else (no unit, empty / unparsable concentration, unsupported unit such as
  nmol/g, no molecular weight) -> ``NaN`` and ``recompute_status = "dropped: <why>"``.
  Nothing is guessed.

The number picked from the text follows DBAASP's own convention (verified: it
reproduces ``activity`` for every ug/ml row): a range or ``<x`` -> its upper bound,
``>x`` -> x, ``v +- e`` -> v. ``direction`` says whether the text is one-sided
(``gt`` / ``lt``) or a point (``exact``); ranges are points at their upper bound.
"""
import html
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from soamp.curation.units import parse_activity

RDLogger.DisableLog("rdApp.*")

NA_SENTINEL = 1_111_111.0
KINDS = ("NA", "empty", "num", "pm", "range", "gt", "lt", "other", "messy", "unparsed")


@dataclass(frozen=True)
class ParsedConc:
    value: float      # the number the rules pick; NaN when there is none
    kind: str         # one of KINDS
    direction: str    # 'gt' | 'lt' | 'exact'


def concentration_direction(text: object) -> str:
    """'gt' for text starting with '>' / '>=' / the >= sign, 'lt' for '<' / '<=' / the <= sign,
    else 'exact'. Same rule as ``soamp.data.dev_dataset.censor_direction`` but applied after
    HTML-unescaping, so ``&gt;128`` is recognised too."""
    s = html.unescape(str(text)).strip()
    if s.startswith((">", "≥")):
        return "gt"
    if s.startswith(("<", "≤")):
        return "lt"
    return "exact"


def parse_concentration(text: object) -> ParsedConc:
    """Free-text concentration -> (value, kind, direction). Rules ported from the checked
    notebook cell (scripts/EDA/09_data_processing.ipynb, section 4c)."""
    direction = concentration_direction(text)
    s = html.unescape(str(text)).strip()
    if s == "NA":
        return ParsedConc(np.nan, "NA", "exact")
    if s == "":
        return ParsedConc(np.nan, "empty", direction)
    if "±" in s or re.fullmatch(r"[\d.,]+\s*\+\s*[\d.,]+", s):   # 9±3, 6.3+0.5 -> the centre
        m = re.search(r"\d+(?:[.,]\d+)?", re.split(r"[±+]", s)[0])   # may carry a prefix, e.g. ">20±5"
        return ParsedConc(float(m.group().replace(",", ".")) if m else np.nan, "pm", direction)
    s = re.sub(r"-\s*>=?", "-", s)                 # "16->32", "4 - >128" are the ranges 16-32, 4-128
    s = s.replace("≤", "<=")                       # parse_activity knows '≥' but not '≤'
    try:
        lo, hi = parse_activity(s)
    except ValueError:                             # control characters etc.
        return ParsedConc(np.nan, "unparsed", direction)
    if np.isnan(lo) and np.isnan(hi):
        return ParsedConc(np.nan, "unparsed", direction)
    value = max(v for v in (lo, hi) if np.isfinite(v))   # upper bound; the lower bound if only '>x'
    if re.fullmatch(r"[\d.,]+", s):
        kind = "num"
    elif np.isinf(hi):
        kind = "gt"
    elif lo == 0 and re.match(r"\s*[<≤]", s):
        kind = "lt"
    elif lo != hi:
        kind = "range"
    else:
        kind = "other"
    if re.search(r"[^\d.,\s<>=≥≤±\-–]", s) or s.count("-") + s.count(">") + s.count("<") > 1:
        kind = "messy"                             # typos such as >>100, 1000>, 13. 8
    return ParsedConc(float(value), kind, direction)


def normalise_unit(unit: object) -> str | None:
    """'ug/ml' | 'uM' | the unit lower-cased as found | None when there is no unit.
    Accepts the micro sign (U+00B5) and the Greek mu (U+03BC) and either spelling of ml."""
    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return None
    u = str(unit).strip().replace("μ", "µ").lower()
    if u in ("", "nan", "none"):
        return None
    if u in ("µg/ml", "ug/ml"):
        return "ug/ml"
    if u in ("µm", "um"):
        return "uM"
    return u


def molecular_weight(smiles: object) -> float:
    """RDKit average molecular weight of a SMILES; NaN if it cannot be parsed."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    return float(Descriptors.MolWt(mol)) if mol is not None else float("nan")


def to_ug_ml(parsed: ParsedConc, unit: object, mw: float) -> tuple[float, str]:
    """(mic_ug_ml, status) for one row; status is 'ok', 'na_sentinel' or 'dropped: <why>'."""
    if parsed.kind == "NA":
        return NA_SENTINEL, "na_sentinel"
    if np.isnan(parsed.value):
        return float("nan"), f"dropped: {'empty' if parsed.kind == 'empty' else 'unparsable'} concentration"
    u = normalise_unit(unit)
    if u is None:
        return float("nan"), "dropped: no unit"
    if u == "ug/ml":
        return parsed.value, "ok"
    if u == "uM":
        if np.isnan(mw):
            return float("nan"), "dropped: no molecular weight"
        return parsed.value * mw / 1000.0, "ok"
    return float("nan"), f"dropped: unsupported unit '{u}'"


def recompute_mic(
    df: pd.DataFrame,
    *,
    smiles_col: str = "smiles",
    id_col: str = "peptide_id",
    conc_col: str = "concentration",
    unit_col: str = "unit",
    activity_col: str = "activity",
) -> pd.DataFrame:
    """Copy of ``df`` with ``conc_value, conc_kind, direction, mw, mic_ug_ml, recompute_status``
    plus audit columns ``activity_dbaasp`` (the original), ``implied_mw`` (the weight DBAASP
    must have used: activity / uM * 1000, uM rows only) and ``activity_ratio``."""
    out = df.copy()
    parsed = [parse_concentration(c) for c in out[conc_col]]
    out["conc_value"] = [p.value for p in parsed]
    out["conc_kind"] = [p.kind for p in parsed]
    out["direction"] = [p.direction for p in parsed]

    mw_by_peptide = {pid: molecular_weight(smi) for pid, smi in
                     out.drop_duplicates(id_col)[[id_col, smiles_col]].itertuples(index=False)}
    out["mw"] = out[id_col].map(mw_by_peptide)

    results = [to_ug_ml(p, u, m) for p, u, m in zip(parsed, out[unit_col], out["mw"])]
    out["mic_ug_ml"] = [r[0] for r in results]
    out["recompute_status"] = [r[1] for r in results]

    out["activity_dbaasp"] = out[activity_col]
    is_um = out[unit_col].map(normalise_unit).eq("uM") & out["conc_value"].gt(0)
    out["implied_mw"] = np.where(is_um, out[activity_col] / out["conc_value"] * 1000.0, np.nan)
    ok = out["recompute_status"].eq("ok") & out["mic_ug_ml"].gt(0)
    out["activity_ratio"] = np.where(ok, out[activity_col] / out["mic_ug_ml"], np.nan)
    return out


_NOT_ACTIVE = re.compile(
    r"(?i)not\s+active\s+(?:up\s*to|at)\s*([\d.,]+)\s*(µg/ml|ug/ml|μg/ml|µ\s*m|μ\s*m|um|micro\s*m|mg/ml|mg/l)")


def parse_not_active_limit(note: object, mw: float) -> float:
    """ug/ml equivalent of a DBAASP note such as 'Not active up to 200 µg/ml' / '... 100 µM'
    (the highest concentration tested without activity), NaN if the note has no such limit.
    Used only to audit NA rows: it says how far 'inactive' is actually supported."""
    m = _NOT_ACTIVE.search(str(note)) if note is not None else None
    if not m:
        return float("nan")
    try:
        number = float(m.group(1).replace(",", ".").rstrip("."))
    except ValueError:
        return float("nan")
    unit = re.sub(r"\s+", "", m.group(2).lower().replace("μ", "µ"))
    if unit in ("µg/ml", "ug/ml", "mg/l"):
        return number
    if unit == "mg/ml":
        return number * 1000.0
    if unit in ("µm", "um", "microm"):
        return number * mw / 1000.0 if not np.isnan(mw) else float("nan")
    return float("nan")
