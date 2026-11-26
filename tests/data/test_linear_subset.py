import pandas as pd
import pytest

from soamp.data.linear_subset import linear_subset, peptide_linearity, smiles_is_linear

LINEAR_TRIPEPTIDE = "CC(C)C[C@H](N)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](C)C(N)=O"  # Leu-Trp-Ala
DISULFIDE = "C(SSC[C@@H](N)C(=O)O)[C@@H](N)C(=O)O"                                        # cystine
MACROCYCLE = "O=C1NCC(=O)NCC(=O)NCC(=O)NCC(=O)NC1"                                        # cyclic pentaglycine


def test_smiles_is_linear_flags_only_backbone_rings():
    assert smiles_is_linear(LINEAR_TRIPEPTIDE) is True   # Trp's 5- and 6-rings are side chains
    assert smiles_is_linear(DISULFIDE) is False
    assert smiles_is_linear(MACROCYCLE) is False
    assert smiles_is_linear("not a smiles") is None


def _dataset(rows):
    return pd.DataFrame([{"peptide_id": pid, "sequence": "AAA", "smiles": smi, "organism": org,
                          "ncbi_taxon_id_if_available": 562, "mic_value_uM": 10.0, "mic_type": "exact",
                          "has_noncanonical": nc, "label": label, "split": split}
                         for pid, smi, nc, org, label, split in rows])


def _split_table(ids):
    return pd.DataFrame([{"peptide_id": pid, "community": 1, "split": "train", "fold_id": 0,
                          "has_noncanonical": False} for pid in ids])


def test_both_sources_must_agree_before_a_peptide_is_kept():
    dataset = _dataset([
        (1, LINEAR_TRIPEPTIDE, False, "Escherichia coli ATCC 25922", "active", "train"),
        (2, DISULFIDE, True, "Escherichia coli ATCC 25922", "inactive", "train"),
        (3, MACROCYCLE, False, "Escherichia coli ATCC 25922", "active", "test"),   # SMILES disagrees
        (4, LINEAR_TRIPEPTIDE, True, "Escherichia coli ATCC 25922", "active", "train"),  # DBAASP disagrees
    ])
    flags = peptide_linearity(dataset).set_index("peptide_id")
    assert list(flags["linear"]) == [True, False, False, False]
    assert flags.at[3, "drop_reason"] == "cyclic_or_disulfide_smiles"
    assert flags.at[4, "drop_reason"] == "dbaasp_intrachain_bond"
    assert flags.at[2, "drop_reason"] == "intrachain_bond_and_cyclic_smiles"


def test_linear_subset_keeps_rows_and_their_split_untouched():
    dataset = _dataset([
        (1, LINEAR_TRIPEPTIDE, False, "Escherichia coli ATCC 25922", "active", "train"),
        (1, LINEAR_TRIPEPTIDE, False, "Staphylococcus aureus ATCC 25923", "inactive", "train"),
        (2, DISULFIDE, True, "Escherichia coli ATCC 25922", "inactive", "test"),
    ])
    result = linear_subset(dataset, _split_table([1, 2]))

    assert list(result.dataset["peptide_id"]) == [1, 1]
    assert list(result.peptide_split["peptide_id"]) == [1]
    # the kept rows are byte-identical to the source rows (no relabelling, no re-splitting)
    pd.testing.assert_frame_equal(result.dataset, dataset.iloc[:2].reset_index(drop=True))
    assert result.stats["pairs_before"] == 3
    assert result.stats["pairs_kept"] == 2
    assert result.stats["peptides_dropped"] == {"intrachain_bond_and_cyclic_smiles": 1}
    assert result.per_strain.at["Escherichia coli ATCC 25922", "pairs_dropped"] == 1
    assert result.per_strain.at["Staphylococcus aureus ATCC 25923", "pairs_dropped"] == 0


def test_a_strain_losing_every_pair_does_not_divide_by_zero():
    dataset = _dataset([(1, DISULFIDE, True, "Bacillus subtilis ATCC 6633", "active", "train")])
    result = linear_subset(dataset, _split_table([1]))
    assert result.dataset.empty
    assert result.per_strain.at["Bacillus subtilis ATCC 6633", "dropped_%"] == 100.0
    assert pd.isna(result.per_strain.at["Bacillus subtilis ATCC 6633", "active_%_kept"])
