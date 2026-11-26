import pandas as pd

from soamp.data.linear_subset import linear_subset, peptide_linearity, smiles_is_linear

LINEAR = "NC(=O)[C@H](CCCCN)NC(=O)[C@H](Cc1ccccc1)N"                      # Phe-Lys-NH2, only 6-membered rings
DISULFIDE = "N[C@H](CSSC[C@H](N)C(=O)O)C(=O)O"                            # cystine
MACROCYCLE = "O=C1CCCCCCCCN1"                                              # 10-membered lactam ring


def _dataset():
    rows = [
        (1, LINEAR, False, "e", "active", "train"),
        (1, LINEAR, False, "s", "inactive", "train"),
        (2, DISULFIDE, False, "e", "active", "test"),     # DBAASP says linear, SMILES has S-S -> dropped
        (3, LINEAR, True, "e", "inactive", "train"),      # DBAASP records a bond -> dropped
        (4, MACROCYCLE, True, "s", "active", "test"),     # both flags -> dropped
    ]
    return pd.DataFrame(rows, columns=["peptide_id", "smiles", "has_noncanonical", "organism", "label", "split"])


def test_smiles_rule():
    assert smiles_is_linear(LINEAR) is True
    assert smiles_is_linear(DISULFIDE) is False
    assert smiles_is_linear(MACROCYCLE) is False
    assert smiles_is_linear("not a smiles") is None


def test_peptide_linearity_reasons():
    reasons = peptide_linearity(_dataset()).set_index("peptide_id")["drop_reason"].to_dict()
    assert reasons == {1: "", 2: "cyclic_or_disulfide_smiles", 3: "dbaasp_intrachain_bond",
                       4: "intrachain_bond_and_cyclic_smiles"}


def test_subset_keeps_only_linear_rows_with_their_original_split():
    dataset = _dataset()
    peptide_split = pd.DataFrame({"peptide_id": [1, 2, 3, 4], "split": ["train", "test", "train", "test"],
                                  "fold_id": [2.0, None, 0.0, None]})
    res = linear_subset(dataset, peptide_split)
    assert res.dataset["peptide_id"].tolist() == [1, 1]
    assert res.peptide_split["peptide_id"].tolist() == [1] and res.peptide_split["fold_id"].tolist() == [2.0]
    assert res.stats["pairs_before"] == 5 and res.stats["pairs_kept"] == 2
    assert res.per_strain.loc["e", "pairs_kept"] == 1
