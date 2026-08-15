import pytest

from soamp.common.thresholds import ThresholdTableError, load_threshold_table, lookup_threshold


def _write_table(tmp_path, rows):
    """rows: list of (match_key, level, active_uM, inactive_uM) -- pass "" for blank."""
    path = tmp_path / "organism_thresholds.csv"
    header = "match_key,level,ncbi_taxon_id_if_available,n_dataset_rows,active_threshold_uM,inactive_threshold_uM,source,notes\n"
    lines = [header]
    for match_key, level, active, inactive in rows:
        lines.append(f'{match_key},{level},,0,{active},{inactive},,\n')
    path.write_text("".join(lines))
    return path


def test_species_match_wins_over_genus(tmp_path):
    path = _write_table(tmp_path, [
        ("Staphylococcus aureus", "species", "32", "128"),
        ("Staphylococcus", "genus", "64", "64"),
    ])
    table = load_threshold_table(path)
    match = lookup_threshold(table, "Staphylococcus aureus")
    assert (match.active_threshold_uM, match.inactive_threshold_uM) == (32, 128)
    assert match.match_level == "species"


def test_genus_fallback_when_no_species_match(tmp_path):
    path = _write_table(tmp_path, [
        ("Pseudomonas", "genus", "16", "64"),
    ])
    table = load_threshold_table(path)
    match = lookup_threshold(table, "Pseudomonas aeruginosa")
    assert (match.active_threshold_uM, match.inactive_threshold_uM) == (16, 64)
    assert match.match_level == "genus"


def test_genus_only_organism_string_matches_genus_row(tmp_path):
    path = _write_table(tmp_path, [
        ("Pseudomonas", "genus", "16", "64"),
    ])
    table = load_threshold_table(path)
    match = lookup_threshold(table, "Pseudomonas sp.")
    assert match.match_level == "genus"


def test_no_match_returns_none(tmp_path):
    path = _write_table(tmp_path, [
        ("Staphylococcus aureus", "species", "32", "128"),
    ])
    table = load_threshold_table(path)
    assert lookup_threshold(table, "Escherichia coli") is None


def test_blank_threshold_treated_as_unmatched(tmp_path):
    path = _write_table(tmp_path, [
        ("Escherichia coli", "species", "", ""),
    ])
    table = load_threshold_table(path)
    assert lookup_threshold(table, "Escherichia coli") is None


def test_equal_active_inactive_collapses_gray_zone(tmp_path):
    path = _write_table(tmp_path, [
        ("Escherichia coli", "species", "32", "32"),
    ])
    table = load_threshold_table(path)
    match = lookup_threshold(table, "Escherichia coli")
    assert (match.active_threshold_uM, match.inactive_threshold_uM) == (32, 32)


def test_duplicate_key_raises(tmp_path):
    path = _write_table(tmp_path, [
        ("Escherichia coli", "species", "16", "64"),
        ("Escherichia coli", "species", "32", "128"),
    ])
    with pytest.raises(ThresholdTableError):
        load_threshold_table(path)


def test_partial_fill_raises(tmp_path):
    path = _write_table(tmp_path, [
        ("Escherichia coli", "species", "32", ""),
    ])
    with pytest.raises(ThresholdTableError):
        load_threshold_table(path)


def test_reversed_breakpoint_raises(tmp_path):
    path = _write_table(tmp_path, [
        ("Escherichia coli", "species", "128", "32"),
    ])
    with pytest.raises(ThresholdTableError):
        load_threshold_table(path)
