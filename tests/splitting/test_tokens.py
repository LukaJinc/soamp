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
