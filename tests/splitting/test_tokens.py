import pytest

from soamp.splitting.tokens import (
    MOD_PREFIX,
    TokenizationError,
    build_modification_table,
    tokenize_linear,
    tokenize_linear_strict,
)


def _entry(position, name):
    return {"position": position, "modificationType": {"name": name, "smiles": "ignored"}}


def test_canonical_only():
    assert tokenize_linear("KWk", []) == ["K", "W", "K"]  # upper-cased


def test_x_becomes_modification_token_at_its_position():
    # peptide 81 in the dev data
    tokens = tokenize_linear("XLWK", [_entry(1, "7-NH2-C7")])
    assert tokens == [MOD_PREFIX + "7-NH2-C7", "L", "W", "K"]


def test_names_used_exactly_as_spelled():
    tokens = tokenize_linear("XXX", [_entry(1, "4-F-PHE"), _entry(2, "4-F-Phe"), _entry(3, "3,4-OH-Arg")])
    assert tokens == ["mod:4-F-PHE", "mod:4-F-Phe", "mod:3,4-OH-Arg"]  # no case-folding / parsing


def test_x_without_entry_is_unresolved():
    assert tokenize_linear("AXA", []) is None
    assert tokenize_linear("AXA", [_entry(1, "Orn")]) is None  # wrong position


def test_conflicting_entries_for_one_position_unresolved():
    assert tokenize_linear("AXA", [_entry(2, "Orn"), _entry(2, "DAB")]) is None


def test_other_characters_unresolved():
    assert tokenize_linear("ABA", []) is None


def test_strict_raises():
    with pytest.raises(TokenizationError):
        tokenize_linear_strict("X", [])


def test_modification_table_counts_sorted():
    table = build_modification_table([["A", "mod:b", "mod:a"], ["mod:b", "K"]])
    assert list(table.items()) == [("mod:b", 2), ("mod:a", 1)]


# ---- d_residues: separate (default) vs map_to_L ---------------------------

from soamp.splitting.tokens import DEFAULT_NAME_EXCEPTIONS, to_l_name  # noqa: E402


def test_default_keeps_d_and_l_names_apart():
    d = tokenize_linear("AXA", [_entry(2, "D-Orn")])
    l = tokenize_linear("AXA", [_entry(2, "Orn")])
    assert d == ["A", "mod:D-Orn", "A"] and l == ["A", "mod:Orn", "A"]
    assert tokenize_linear("AXA", [_entry(2, "D-Orn")], "separate") == d


def test_map_to_l_merges_d_and_l_forms():
    d = tokenize_linear("AxA", [_entry(2, "D-Orn")], "map_to_L")
    l = tokenize_linear("AXA", [_entry(2, "Orn")], "map_to_L")
    assert d == l == ["A", "mod:Orn", "A"]


@pytest.mark.parametrize("raw,expected", [
    ("D-Orn", "Orn"), ("d-orn", "orn"), ("D-Allo-Ile", "Allo-Ile"),
    ("D-1-NAL", "1-NAL"), ("D-Lys-C12", "Lys-C12"),
    ("D-TIC", "Tic"),                                   # case collision: explicit table
    ("4-NH2-D-Pro", "4-NH2-Pro"), ("cis-4-D-Hyp", "cis-4-Hyp"),
    ("C10-N6-D-Lys", "C10-N6-Lys"), ("β-Thi-D-Ala", "β-Thi-Ala"),
])
def test_to_l_name_cases(raw, expected):
    assert to_l_name(raw) == expected


@pytest.mark.parametrize("name", [
    "Orn", "NLys", "DAB", "DAP", "Dab(Cys)",          # starts with D but not 'D-'
    "(S,S)-ACPC", "(R,R)-ACPC", "trans-4-Hyp", "Allo-Ile",   # descriptors that are not D/L
])
def test_to_l_name_leaves_other_names_alone(name):
    assert to_l_name(name) == name


def test_map_to_l_does_not_change_canonical_or_unmarked():
    assert tokenize_linear("kKW", [], "map_to_L") == tokenize_linear("kKW", [], "separate") == ["K", "K", "W"]
    assert tokenize_linear("XLWK", [_entry(1, "7-NH2-C7")], "map_to_L") == [MOD_PREFIX + "7-NH2-C7", "L", "W", "K"]


def test_map_to_l_position_conflict_is_checked_after_mapping():
    assert tokenize_linear("AX", [_entry(2, "D-Orn"), _entry(2, "Orn")], "map_to_L") == ["A", "mod:Orn"]
    assert tokenize_linear("AX", [_entry(2, "D-Orn"), _entry(2, "Orn")], "separate") is None


def test_custom_exception_table_and_bad_option():
    assert to_l_name("D-TIC", {}) == "TIC"
    assert "D-TIC" in DEFAULT_NAME_EXCEPTIONS
    with pytest.raises(ValueError):
        tokenize_linear("AXA", [_entry(2, "D-Orn")], "mirror")
