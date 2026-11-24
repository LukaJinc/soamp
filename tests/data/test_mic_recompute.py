import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soamp.data.mic_recompute import (
    NA_SENTINEL,
    concentration_direction,
    molecular_weight,
    normalise_unit,
    parse_concentration,
    recompute_mic,
    to_ug_ml,
)

GLY = "NCC(=O)O"   # glycine, MolWt 75.07


@pytest.mark.parametrize("text,value,kind,direction", [
    ("8", 8.0, "num", "exact"),
    ("1,5", 1.5, "num", "exact"),
    (">128", 128.0, "gt", "gt"),
    (">=64", 64.0, "gt", "gt"),
    ("≥4", 4.0, "gt", "gt"),
    ("<2", 2.0, "lt", "lt"),
    ("<=1", 1.0, "lt", "lt"),
    ("≤1", 1.0, "lt", "lt"),
    ("4-128", 128.0, "range", "exact"),          # a range is a point at its upper bound
    ("16->32", 32.0, "range", "exact"),
    ("4 - >128", 128.0, "range", "exact"),
    ("9±3", 9.0, "pm", "exact"),
    ("6.3+0.5", 6.3, "pm", "exact"),
    (">20±5", 20.0, "pm", "gt"),
    ("&gt;128", 128.0, "gt", "gt"),               # HTML-escaped censoring is recognised
    ("&lt;2", 2.0, "lt", "lt"),
])
def test_parse_concentration_values(text, value, kind, direction):
    p = parse_concentration(text)
    assert (p.value, p.kind, p.direction) == (value, kind, direction)


def test_na_empty_and_unparsable():
    na = parse_concentration("NA")
    assert na.kind == "NA" and np.isnan(na.value) and na.direction == "exact"
    assert parse_concentration("").kind == "empty"
    assert parse_concentration("   ").kind == "empty"
    for bad in ("abc", None, float("nan")):
        p = parse_concentration(bad)
        assert p.kind == "unparsed" and np.isnan(p.value)


def test_messy_typos_still_give_a_value_but_are_flagged():
    p = parse_concentration(">>100")
    assert p.kind == "messy" and p.value == 100.0 and p.direction == "gt"


def test_direction_matches_the_old_censor_rule_on_plain_text():
    from soamp.data.dev_dataset import censor_direction
    for t in (">128", ">=0.6", "≥4", "<2", "≤1", "8", "NA", "4-128"):
        assert concentration_direction(t) == censor_direction(t)


@pytest.mark.parametrize("raw,expected", [
    ("µM", "uM"), ("μM", "uM"), ("uM", "uM"), ("µg/ml", "ug/ml"), ("μg/ml", "ug/ml"), ("ug/ml", "ug/ml"),
    ("nmol/g", "nmol/g"), (None, None), (float("nan"), None), ("", None),
])
def test_normalise_unit(raw, expected):
    assert normalise_unit(raw) == expected


def test_conversion_rules():
    mw = molecular_weight(GLY)
    assert math.isclose(mw, 75.067, abs_tol=0.01)
    assert to_ug_ml(parse_concentration("10"), "µg/ml", mw) == (10.0, "ok")                  # passthrough
    value, status = to_ug_ml(parse_concentration("100"), "µM", mw)                          # uM * MW / 1000
    assert status == "ok" and math.isclose(value, 100 * mw / 1000)
    assert to_ug_ml(parse_concentration("NA"), "µM", mw) == (NA_SENTINEL, "na_sentinel")     # any unit
    assert to_ug_ml(parse_concentration("NA"), None, float("nan")) == (NA_SENTINEL, "na_sentinel")
    assert NA_SENTINEL == 1_111_111.0


@pytest.mark.parametrize("conc,unit,why", [
    ("", "µM", "empty concentration"),
    ("5", None, "no unit"),
    ("5", float("nan"), "no unit"),
    ("5", "nmol/g", "unsupported unit 'nmol/g'"),
    ("abc", "µg/ml", "unparsable concentration"),
])
def test_unrecomputable_rows_are_dropped_with_a_reason(conc, unit, why):
    value, status = to_ug_ml(parse_concentration(conc), unit, 75.0)
    assert np.isnan(value) and status == f"dropped: {why}"


def test_uM_row_without_molecular_weight_is_dropped():
    value, status = to_ug_ml(parse_concentration("5"), "µM", float("nan"))
    assert np.isnan(value) and status == "dropped: no molecular weight"


def test_recompute_mic_table_and_audit_columns():
    df = pd.DataFrame({
        "peptide_id": [1, 1, 2, 3, 4],
        "smiles": [GLY, GLY, GLY, "not a smiles", GLY],
        "concentration": ["100", ">64", "NA", "5", "7"],
        "unit": ["µM", "µg/ml", None, "µM", None],
        # DBAASP's stored activity: row 0 used a bogus MW of 17.03 instead of 75.07
        "activity": [100 * 17.03 / 1000, 64.0, 1111111.0, 99.0, 7.0],
    })
    out = recompute_mic(df)
    mw = molecular_weight(GLY)
    assert math.isclose(out.loc[0, "mic_ug_ml"], 100 * mw / 1000)
    assert out.loc[1, "mic_ug_ml"] == 64.0 and out.loc[1, "direction"] == "gt"
    assert out.loc[2, "mic_ug_ml"] == NA_SENTINEL and out.loc[2, "recompute_status"] == "na_sentinel"
    assert out.loc[3, "recompute_status"] == "dropped: no molecular weight"
    assert out.loc[4, "recompute_status"] == "dropped: no unit"
    assert math.isclose(out.loc[0, "implied_mw"], 17.03)               # exposes the bogus weight
    assert out.loc[0, "activity_dbaasp"] == df.loc[0, "activity"]       # original kept
    assert out.loc[1, "activity_ratio"] == 1.0
    assert len(out) == len(df) and "mic_ug_ml" not in df.columns        # input untouched


REAL = Path(__file__).resolve().parents[2] / "data/dev/dbaasp_dev_processed.parquet"
SEVEN = ["Escherichia coli ATCC 25922", "Staphylococcus aureus ATCC 25923", "Pseudomonas aeruginosa ATCC 27853",
         "Acinetobacter baumannii ATCC 19606", "Klebsiella pneumoniae ATCC 700603",
         "Enterococcus faecalis ATCC 29212", "Bacillus subtilis ATCC 6633"]


@pytest.mark.skipif(not REAL.exists(), reason="dev processed parquet not available")
def test_real_data_regression_guard_seven_strains():
    """Locks what was measured on the real data: exactly 3 rows cannot be recomputed
    (1 empty uM, 2 plain numbers without a unit) and all 1,113 NA rows get the sentinel."""
    df = pd.read_parquet(REAL)
    df = df[df["target_species"].isin(SEVEN)]
    out = recompute_mic(df)
    dropped = out[out["recompute_status"].str.startswith("dropped")]
    assert len(out) == 30568
    assert dropped["recompute_status"].value_counts().to_dict() == {
        "dropped: no unit": 2, "dropped: empty concentration": 1}
    na = out[out["conc_kind"] == "NA"]
    assert len(na) == 1113 and (na["mic_ug_ml"] == NA_SENTINEL).all()
    ugml = out[(out["unit"] == "µg/ml") & (out["conc_kind"] == "num")]
    assert ((ugml["activity_ratio"] - 1).abs() < 1e-3).mean() > 0.999   # ug/ml rows match DBAASP
