import json

from soamp.labeling.censoring import recover_bounds

PEPTIDES = [
    {
        "id": 90001,
        "targetActivities": [
            {
                "activityMeasureGroup": {"name": "MIC"},
                "targetSpecies": {"name": "Staphylococcus aureus"},
                "unit": {"name": "µM"},
                "concentration": "<0.5",
            },
            {
                # non-MIC assay type -- must be filtered out
                "activityMeasureGroup": {"name": "MIC50"},
                "targetSpecies": {"name": "Staphylococcus aureus"},
                "unit": {"name": "µM"},
                "concentration": "1",
            },
        ],
    },
    {
        "id": 90002,
        "targetActivities": [
            {
                "activityMeasureGroup": {"name": "MIC"},
                "targetSpecies": {"name": "Escherichia coli"},
                "unit": {"name": "µM"},
                "concentration": ">100",
            },
        ],
    },
    {
        "id": 90003,
        "targetActivities": [
            {
                "activityMeasureGroup": {"name": "MIC"},
                "targetSpecies": {"name": "Bacillus subtilis"},
                "unit": {"name": "µM"},
                "concentration": "4-8",
            },
        ],
    },
]


def _write_fixture(tmp_path):
    path = tmp_path / "dbaasp_raw.jsonl"
    with open(path, "w") as f:
        for p in PEPTIDES:
            f.write(json.dumps(p) + "\n")
    return path


def test_left_censored_recovered_as_upper_bound(tmp_path):
    path = _write_fixture(tmp_path)
    pairs = {(90001, "Staphylococcus aureus")}
    result = recover_bounds(path, pairs, molecular_weights={})
    b = result[(90001, "Staphylococcus aureus")]
    assert b.status == "recovered"
    assert b.censor_type == "censored"
    assert b.min_uM == 0.0
    assert b.max_uM == 0.5


def test_right_censored_recovered_as_lower_bound(tmp_path):
    path = _write_fixture(tmp_path)
    pairs = {(90002, "Escherichia coli")}
    result = recover_bounds(path, pairs, molecular_weights={})
    b = result[(90002, "Escherichia coli")]
    assert b.status == "recovered"
    assert b.censor_type == "censored"
    assert b.min_uM == 100.0
    assert b.max_uM == float("inf")


def test_two_sided_range_recovered_as_ranged(tmp_path):
    path = _write_fixture(tmp_path)
    pairs = {(90003, "Bacillus subtilis")}
    result = recover_bounds(path, pairs, molecular_weights={})
    b = result[(90003, "Bacillus subtilis")]
    assert b.status == "recovered"
    assert b.censor_type == "ranged"
    assert b.min_uM == 4.0
    assert b.max_uM == 8.0


def test_pair_with_no_matching_raw_measurement_is_zero_matches(tmp_path):
    path = _write_fixture(tmp_path)
    pairs = {(99999, "Nonexistent organism")}
    result = recover_bounds(path, pairs, molecular_weights={})
    b = result[(99999, "Nonexistent organism")]
    assert b.status == "zero_matches"
    assert b.min_uM == 0.0
    assert b.max_uM == float("inf")


def test_non_mic_assay_group_is_excluded(tmp_path):
    # peptide 90001's second targetActivity (MIC50) must not contaminate the
    # single recovered measurement for its MIC record.
    path = _write_fixture(tmp_path)
    pairs = {(90001, "Staphylococcus aureus")}
    result = recover_bounds(path, pairs, molecular_weights={})
    assert result[(90001, "Staphylococcus aureus")].status == "recovered"
