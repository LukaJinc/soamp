from soamp.stereo.notation import parse_notation
from soamp.stereo.twins import to_l_token, twin_pairs
from soamp.stereo.audit import peptide_label


def _u(pos, name, smiles=None):
    return {"position": pos, "modificationType": {"name": name, "smiles": smiles}}


def test_canonical_case_and_glycine():
    rows = parse_notation("kKg", None)
    assert [r["notation_stereo"] for r in rows] == ["D", "L", "achiral"]
    assert all(r["residue_class"] == "standard" for r in rows)


def test_unusual_case_and_name_agree():
    rows = parse_notation("AxA", [_u(2, "D-ORN")])
    assert rows[1]["residue_class"] == "nonstandard"
    assert (rows[1]["case_stereo"], rows[1]["name_stereo"], rows[1]["notation_stereo"]) == ("D", "D", "D")


def test_unusual_case_and_name_disagree_is_undetermined_not_picked():
    row = parse_notation("AXA", [_u(2, "D-ORN")])[1]
    assert (row["case_stereo"], row["name_stereo"]) == ("L", "D")
    assert row["notation_stereo"] == "undetermined" and row["notation_reason"]


def test_unresolved_x_is_unrecognized():
    assert parse_notation("AXA", None)[1]["residue_class"] == "unrecognized"


def test_to_l_token():
    assert to_l_token("K", "k") == "K"
    assert to_l_token("D-ORN", "x") == "ORN"
    assert to_l_token("ORN", "X") == "ORN"
    assert to_l_token("NLys", "X") == "NLys"      # 'N' prefix is not 'D-'


def test_twin_pairs():
    recs = [
        {"peptide_id": 1, "mapped": ("K", "L"), "stereo": ("L", "L"), "annotation": "a"},
        {"peptide_id": 2, "mapped": ("K", "L"), "stereo": ("D", "L"), "annotation": "a"},
        {"peptide_id": 3, "mapped": ("K", "L"), "stereo": ("D", "L"), "annotation": "b"},
        {"peptide_id": 4, "mapped": ("K", "V"), "stereo": ("L", "L"), "annotation": "a"},
    ]
    pairs = {(p["peptide_a"], p["peptide_b"]): p["same_annotation"] for p in twin_pairs(recs)}
    assert pairs == {(1, 2): True, (1, 3): False}   # 2 and 3 share stereo, so they are not stereo twins


def test_peptide_label():
    assert peptide_label(["L", "achiral"]) == "all_L"
    assert peptide_label(["D", "D"]) == "all_D"
    assert peptide_label(["L", "D"]) == "mixed"
    assert peptide_label(["achiral"]) == "no_chiral_residues"
    assert peptide_label(["L", "undetermined"]) == "undetermined"
    assert peptide_label(["D", "undetermined"], ignore_undetermined=True) == "all_D"
