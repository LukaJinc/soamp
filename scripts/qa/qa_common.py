"""
Shared context loader and helpers for the independent QA/validation pass over the
curated AMP MIC regression dataset (scripts/01-08 pipeline outputs).

This package is deliberately NOT part of the numbered pipeline (scripts/01-08) --
it is an external auditor. Where a checklist item asks for *independent*
verification, the helpers here are hand-rolled from raw data rather than calling
back into scripts/common/units.py or the assay/filter logic in
scripts/06_assay_filter_units.py, so a bug in that logic would actually be caught.
The two exceptions (disclosed, not hidden) are:
  - scripts/common/parse_dbaasp.py::DBAASPPeptide -- pure JSON flattening, makes
    no inclusion/exclusion or arithmetic decisions.
  - scripts/common/taxonomy.py::classify_species -- wraps ete4's NCBI taxonomy
    lookup; reimplementing NCBI lineage resolution is out of scope, and a bug
    there would be a bug in ete4/NCBI data, not in this pipeline. What IS
    independently checked (Section 6) is whether the pipeline actually *applied*
    the Bacteria-domain filter to every row, using this same classifier.
"""
import os
import sys
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.*')

COMMON_DIR = os.path.join(os.path.dirname(__file__), "..", "common")
sys.path.insert(0, os.path.abspath(COMMON_DIR))
from taxonomy import classify_species  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(BASE, "data")
REPORTS = os.path.join(BASE, "reports")
CACHE = os.path.join(BASE, ".cache")


@dataclass
class SectionResult:
    section_id: str
    title: str
    status: str  # "PASS" | "FAIL" | "FLAGGED"
    metrics: dict = field(default_factory=dict)
    tables: list = field(default_factory=list)   # list of (caption, list[dict])
    narrative: list = field(default_factory=list)  # list of markdown lines
    follow_ups: list = field(default_factory=list)


@dataclass
class QAContext:
    raw_by_id: dict            # str(peptide_id) -> raw DBAASP JSON dict
    qmap_raw: list              # freshly reparsed .cache/qmap_hf/dbaasp.json
    qmap_included: pd.DataFrame
    dbaasp_raw_full: pd.DataFrame
    diff: pd.DataFrame
    recovered: pd.DataFrame
    unconvertible: pd.DataFrame
    step5: pd.DataFrame
    final: pd.DataFrame
    split: dict
    audit_text: str
    results: dict = field(default_factory=dict)   # populated as sections run


def _read_csv(path, **kw):
    return pd.read_csv(path, dtype={"peptide_id": str}, keep_default_na=True, **kw)


def load_context() -> QAContext:
    print("Loading raw DBAASP jsonl ...", flush=True)
    raw_by_id = {}
    with open(os.path.join(CACHE, "dbaasp_raw.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            raw_by_id[str(d["id"])] = d
    print(f"  {len(raw_by_id)} raw peptide records loaded", flush=True)

    print("Reparsing QMAP's original HF source (dbaasp.json) fresh ...", flush=True)
    with open(os.path.join(CACHE, "qmap_hf", "dbaasp.json")) as f:
        qmap_raw = json.load(f)
    print(f"  {len(qmap_raw)} QMAP records loaded", flush=True)

    qmap_included = _read_csv(os.path.join(DATA, "qmap_included.csv"))
    dbaasp_raw_full = _read_csv(os.path.join(DATA, "dbaasp_raw_full.csv"))
    diff = _read_csv(os.path.join(DATA, "dbaasp_vs_qmap_diff.csv"))
    recovered = _read_csv(os.path.join(DATA, "recovered_peptides.csv"))
    unconvertible = _read_csv(os.path.join(DATA, "unconvertible_peptides.csv"))
    step5 = _read_csv(os.path.join(DATA, "step5_standardized_mic.csv"))
    final = _read_csv(os.path.join(DATA, "final_mic_regression_dataset.csv"))

    with open(os.path.join(DATA, "split_indices.json")) as f:
        split = json.load(f)
    split["train_peptide_ids"] = [str(x) for x in split["train_peptide_ids"]]
    split["test_peptide_ids"] = [str(x) for x in split["test_peptide_ids"]]

    with open(os.path.join(REPORTS, "curation_audit.md")) as f:
        audit_text = f.read()

    return QAContext(
        raw_by_id=raw_by_id, qmap_raw=qmap_raw, qmap_included=qmap_included,
        dbaasp_raw_full=dbaasp_raw_full, diff=diff, recovered=recovered,
        unconvertible=unconvertible, step5=step5, final=final, split=split,
        audit_text=audit_text,
    )


# ---------------------------------------------------------------------------
# Hand-rolled, independent helpers (deliberately not importing scripts/common/units.py)
# ---------------------------------------------------------------------------

def rdkit_mw(smiles: str) -> Optional[float]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)


def ug_ml_to_uM_by_hand(ug_ml: float, mw: float) -> float:
    """uM = 1000 * (ug/mL) / (g/mol) -- literal formula, no helper indirection."""
    return 1000.0 * ug_ml / mw


def rdkit_roundtrip_stable(smiles: str):
    """Returns (parseable: bool, roundtrip_stable: bool, canonical_smiles: str|None)."""
    m1 = Chem.MolFromSmiles(smiles)
    if m1 is None:
        return False, False, None
    c1 = Chem.MolToSmiles(m1, canonical=True)
    m2 = Chem.MolFromSmiles(c1)
    if m2 is None:
        return True, False, c1
    c2 = Chem.MolToSmiles(m2, canonical=True)
    return True, (c1 == c2), c1


_NUMERIC_RE = re.compile(r"^\s*\d+(\.\d+)?\s*$")


def is_plain_numeric(s: str) -> bool:
    return bool(s) and bool(_NUMERIC_RE.match(s))


def build_raw_activity_index(ctx: QAContext, peptide_ids):
    """
    One pass over ctx.raw_by_id restricted to peptide_ids, pre-extracting per
    targetActivity record: {assay_group, species_name, domain, binomial, unit,
    concentration_raw}. assay_group=='MIC' and domain=='Bacteria' are the two
    conditions Section 6 checks -- written as plain field comparisons here, not
    imported from 06_assay_filter_units.py.

    Returns: dict[str peptide_id] -> list[dict]
    """
    idx = {}
    species_cache = {}
    for pid in peptide_ids:
        raw = ctx.raw_by_id.get(pid)
        if raw is None:
            idx[pid] = []
            continue
        activities = []
        for t in raw.get("targetActivities") or []:
            group = (t.get("activityMeasureGroup") or {}).get("name")
            species_name = (t.get("targetSpecies") or {}).get("name")
            if species_name not in species_cache:
                species_cache[species_name] = classify_species(species_name) if species_name else ("Unknown", None, None)
            domain, taxid, binomial = species_cache[species_name]
            unit = (t.get("unit") or {}).get("name")
            activities.append({
                "assay_group": group,
                "species_name": species_name,
                "domain": domain,
                "binomial": binomial,
                "unit": unit,
                "concentration_raw": t.get("concentration"),
            })
        idx[pid] = activities
    return idx


def fmt_int(n) -> str:
    return f"{n:,}"


def md_table(rows: list, columns: Optional[list] = None) -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = columns or list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"
