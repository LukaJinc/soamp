

def test_build_threshold_report_resolves_strains_and_records_unit():
    from soamp.common.thresholds import ThresholdTable, build_threshold_report

    table = ThresholdTable(species={"Escherichia coli": (32.0, 128.0)}, genus={"Bacillus": (8.0, 64.0)})
    rows = build_threshold_report(
        ["Escherichia coli ATCC 25922", "Bacillus cereus ATCC 11778", "Mystery sp. X"],
        table, "µg/ml", {("species", "Escherichia coli"): "user-specified"},
    )
    by = {r["organism"]: r for r in rows}
    ecoli = by["Escherichia coli ATCC 25922"]
    assert (ecoli["match_level"], ecoli["active_threshold"], ecoli["inactive_threshold"]) == ("species", 32.0, 128.0)
    assert ecoli["threshold_unit"] == "µg/ml" and ecoli["source"] == "user-specified"
    assert by["Bacillus cereus ATCC 11778"]["match_level"] == "genus"
    assert by["Mystery sp. X"]["match_level"] == "none" and by["Mystery sp. X"]["active_threshold"] is None
