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
