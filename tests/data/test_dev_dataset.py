import math

import pandas as pd
import pytest

from soamp.common.thresholds import ThresholdTable
from soamp.data.dev_dataset import (
    DevDatasetConfig,
    DevDatasetError,
    build_dev_dataset,
    censor_direction,
    label_record,
    merge_labels,
    species_of,
)

ORG = "Escherichia coli ATCC 25922"
TABLE = ThresholdTable(species={"Escherichia coli": (32.0, 128.0)}, genus={})
CFG = DevDatasetConfig(target_organisms=[ORG])


@pytest.mark.parametrize("raw,expected", [
    (">128", "gt"), (">=0.6", "gt"), ("≥4", "gt"), ("<2", "lt"), ("≤1", "lt"), ("8", "exact"), ("NA", "exact"),
])
def test_censor_direction(raw, expected):
    assert censor_direction(raw) == expected


def test_species_of_strips_strain():
    assert species_of("Escherichia coli ATCC 25922") == "Escherichia coli"


@pytest.mark.parametrize("act,direction,expected", [
    (16, "exact", "active"), (64, "exact", "uncertain"), (200, "exact", "inactive"),
    (8, "gt", "uncertain"),      # '>8' could be anything >= 8: not active
    (200, "gt", "inactive"),
    (4, "lt", "active"), (64, "lt", "uncertain"),
])
def test_label_record(act, direction, expected):
    assert label_record(act, direction, 32.0, 128.0) == expected


def test_label_record_without_censoring_treats_value_as_exact():
    assert label_record(8, "gt", 32.0, 128.0, use_censoring=False) == "active"


def test_merge_labels():
    assert merge_labels(["active", "uncertain", "active"]) == "active"
    assert merge_labels(["active", "inactive"]) is None
    assert merge_labels(["uncertain"]) is None


def _processed(rows):
    return pd.DataFrame([{
        "peptide_id": pid, "sequence_full": "AK", "smiles": "CC", "is_linear": lin, "target_species": ORG,
        "measure": "MIC", "activity": act, "concentration": conc,
    } for pid, lin, act, conc in rows])


def test_build_dev_dataset_labels_drops_and_splits():
    processed = _processed([
        (1, True, 8.0, "8"), (1, True, 16.0, "16"),      # active (merged)
        (2, False, 256.0, ">256"),                        # inactive, nonlinear
        (3, True, 8.0, ">8"),                             # uncertain -> dropped
        (4, True, 8.0, "8"), (4, True, 256.0, "256"),     # conflict -> dropped
    ])
    split = pd.DataFrame({"peptide_id": [1, 2, 3, 4], "cluster_id": ["a1", "m1", "a2", "a3"],
                          "split": ["train", "train", "test", "train"], "fold_id": [0.0, 1.0, float("nan"), 2.0]})
    res = build_dev_dataset(processed, split, TABLE, CFG)
    assert res.dataset.set_index("peptide_id")["label"].to_dict() == {1: "active", 2: "inactive"}
    assert res.dataset.set_index("peptide_id")["has_noncanonical"].to_dict() == {1: False, 2: True}
    assert res.peptide_split["fold_id"].tolist() == ["0", "1"]
    assert math.isclose(res.dataset.set_index("peptide_id").loc[1, "mic_value_uM"], math.sqrt(8 * 16))


def test_build_dev_dataset_requires_split_entry():
    processed = _processed([(1, True, 8.0, "8")])
    split = pd.DataFrame({"peptide_id": [2], "cluster_id": ["a"], "split": ["train"], "fold_id": [0.0]})
    with pytest.raises(DevDatasetError, match="no split entry"):
        build_dev_dataset(processed, split, TABLE, CFG)


def test_build_dev_dataset_requires_threshold():
    with pytest.raises(DevDatasetError, match="no threshold"):
        build_dev_dataset(_processed([(1, True, 8.0, "8")]), pd.DataFrame(), ThresholdTable({}, {}), CFG)


SPLIT1 = pd.DataFrame({"peptide_id": [1], "cluster_id": ["a"], "split": ["train"], "fold_id": [0.0]})
DBAASP_TABLE = ThresholdTable(species={"Escherichia coli": (25.0, 100.0)}, genus={})


def _label_of(activity, strict=True, smiles="C"):
    cfg = DevDatasetConfig(target_organisms=[ORG], strict_boundaries=strict)
    processed = _processed([(1, True, activity, str(activity))]).assign(smiles=smiles)
    try:
        return build_dev_dataset(processed, SPLIT1, DBAASP_TABLE, cfg).dataset["label"].tolist()[0]
    except DevDatasetError:
        return "dropped"


@pytest.mark.parametrize("activity,expected", [
    (24.9, "active"), (25.0, "dropped"), (60.0, "dropped"), (100.0, "dropped"), (100.1, "inactive"),
])
def test_dbaasp_25_100_strict_boundaries(activity, expected):
    assert _label_of(activity) == expected


@pytest.mark.parametrize("activity,expected", [(25.0, "active"), (60.0, "dropped"), (100.0, "inactive")])
def test_inclusive_boundaries_when_not_strict(activity, expected):
    assert _label_of(activity, strict=False) == expected


def test_thresholds_apply_directly_to_activity_without_mw_conversion():
    """Locks the decision: cutoffs apply to `activity` as stored (ug/ml-normalised),
    whatever the peptide's molecular weight."""
    heavy = "CC(C)C[C@H](NC(=O)[C@H](C)N)C(=O)O" * 5
    assert _label_of(24.0, smiles="C") == _label_of(24.0, smiles=heavy) == "active"
    assert _label_of(60.0, smiles="C") == _label_of(60.0, smiles=heavy) == "dropped"


# ---- dev v2: recomputed value column, NA rows, label stage without a split ---------------

from soamp.data.dev_dataset import attach_split, label_dev_pairs  # noqa: E402
from soamp.data.mic_recompute import NA_SENTINEL  # noqa: E402

V2_TABLE = ThresholdTable(species={"Escherichia coli": (25.0, 100.0)}, genus={})
V2_CFG = DevDatasetConfig(target_organisms=[ORG], value_column="mic_ug_ml", direction_column="direction",
                          include_na_rows=True)


def _v2(rows):
    """rows: (peptide_id, is_linear, measure, concentration, mic_ug_ml, direction, conc_kind, note)"""
    return pd.DataFrame([{
        "peptide_id": pid, "sequence_full": "AK", "smiles": "CC", "is_linear": lin, "target_species": ORG,
        "measure": measure, "concentration": conc, "mic_ug_ml": val, "direction": d, "conc_kind": kind,
        "activity": 12345.0,       # DBAASP's value must be ignored when value_column is mic_ug_ml
        "note": note, "mw": 1000.0,
    } for pid, lin, measure, conc, val, d, kind, note in rows])


def _labels(rows, cfg=V2_CFG):
    res = label_dev_pairs(_v2(rows), V2_TABLE, cfg)
    return res, res.dataset.set_index("peptide_id")["label"].to_dict()


def test_default_config_is_the_old_behaviour():
    cfg = DevDatasetConfig()
    assert (cfg.value_column, cfg.direction_column, cfg.include_na_rows) == ("activity", None, False)
    assert cfg.paths.cluster_parquet is None


def test_value_column_replaces_dbaasp_activity():
    _, lab = _labels([(1, True, "MIC", "5", 10.0, "exact", "num", ""), (2, True, "MIC", "5", 500.0, "exact", "num", "")])
    assert lab == {1: "active", 2: "inactive"}      # activity=12345 would have made both inactive


def test_na_row_is_inactive_and_nothing_else_admits_other_measures():
    rows = [(1, True, "-", "NA", NA_SENTINEL, "exact", "NA", "Not active up to 200 µg/ml"),
            (2, True, "MBC", "5", 1.0, "exact", "num", ""),                   # non-MIC, non-NA: still excluded
            (3, True, "MIC", "5", 10.0, "exact", "num", "")]
    res, lab = _labels(rows)
    assert lab == {1: "inactive", 3: "active"}
    ds = res.dataset.set_index("peptide_id")
    assert ds.loc[1, "mic_type"] == "na" and ds.loc[1, "mic_value_uM"] == NA_SENTINEL
    assert list(res.na_only["peptide_id"]) == [1] and res.stats["na_only_labelled_pairs"] == 1
    assert res.na_only.iloc[0]["not_active_up_to_ug_ml"] == 200.0


def test_na_rows_ignored_when_not_included():
    rows = [(1, True, "-", "NA", NA_SENTINEL, "exact", "NA", ""), (3, True, "MIC", "5", 10.0, "exact", "num", "")]
    cfg = DevDatasetConfig(target_organisms=[ORG], value_column="mic_ug_ml", direction_column="direction")
    _, lab = _labels(rows, cfg)
    assert lab == {3: "active"}


def test_na_conflicting_with_an_active_mic_drops_the_pair_and_is_reported():
    rows = [(1, True, "MIC", "5", 10.0, "exact", "num", ""),                 # active
            (1, True, "-", "NA", NA_SENTINEL, "exact", "NA", ""),            # inactive
            (2, True, "MIC", "500", 500.0, "exact", "num", ""),              # inactive
            (2, True, "-", "NA", NA_SENTINEL, "exact", "NA", ""),            # inactive: agrees
            (3, True, "MIC", "60", 60.0, "exact", "num", ""),                # uncertain
            (3, True, "-", "NA", NA_SENTINEL, "exact", "NA", "")]            # -> pair becomes inactive
    res, lab = _labels(rows)
    assert lab == {2: "inactive", 3: "inactive"}
    assert res.conflicts[["peptide_id", "involves_na_row"]].values.tolist() == [[1, True]]
    assert res.stats["of_which_involving_an_na_row"] == 1


def test_mic_value_ignores_the_sentinel_when_real_values_exist():
    rows = [(2, True, "MIC", "400", 400.0, "exact", "num", ""), (2, True, "MIC", "900", 900.0, "exact", "num", ""),
            (2, True, "-", "NA", NA_SENTINEL, "exact", "NA", "")]
    res, _ = _labels(rows)
    ds = res.dataset.set_index("peptide_id")
    assert math.isclose(ds.loc[2, "mic_value_uM"], math.sqrt(400 * 900)) and ds.loc[2, "mic_type"] == "averaged"


def test_censoring_uses_the_direction_column():
    rows = [(1, True, "MIC", ">8", 8.0, "gt", "gt", ""),       # >8 could be anything above: uncertain -> dropped
            (2, True, "MIC", ">200", 200.0, "gt", "gt", ""),   # inactive
            (3, True, "MIC", "<10", 10.0, "lt", "lt", "")]     # active
    _, lab = _labels(rows)
    assert lab == {2: "inactive", 3: "active"}


@pytest.mark.parametrize("value,expected", [(24.9, "active"), (25.0, None), (100.0, None), (100.1, "inactive")])
def test_v2_strict_25_100(value, expected):
    try:
        _, lab = _labels([(1, True, "MIC", str(value), value, "exact", "num", "")])
        got = lab.get(1)
    except DevDatasetError:
        got = None
    assert got == expected


def test_attach_split_matches_build_dev_dataset():
    processed = _processed([(1, True, 8.0, "8"), (2, False, 256.0, ">256")])
    split = pd.DataFrame({"peptide_id": [1, 2], "cluster_id": ["a1", "m1"], "split": ["train", "test"],
                          "fold_id": [0.0, float("nan")]})
    labelled = label_dev_pairs(processed, TABLE, CFG)
    assert "split" not in labelled.dataset.columns
    ds, ps = attach_split(labelled.dataset, split)
    ref = build_dev_dataset(processed, split, TABLE, CFG)
    assert ds.equals(ref.dataset) and ps.equals(ref.peptide_split)
