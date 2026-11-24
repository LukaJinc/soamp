import pytest
from rdkit import Chem

from soamp.stereo.match import (
    alpha_sites, chain_order, corn_stereo, match_template, standard_templates, template_stereo,
)


def _stereo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return corn_stereo(mol, alpha_sites(mol)[0])[0]


def _standard_hit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    site = alpha_sites(mol)[0]
    return {k: template_stereo(t, rel) for k, t in standard_templates().items()
            if (rel := match_template(mol, site, t))}


@pytest.mark.parametrize("smiles,expected", [
    ("C[C@H](N)C(=O)O", "L"),                 # L-Ala
    ("C[C@@H](N)C(=O)O", "D"),                # D-Ala
    ("C([C@@H](C(=O)O)N)S", "L"),             # L-Cys: R by CIP, L by convention
    ("C([C@H](C(=O)O)N)S", "D"),              # D-Cys
    ("NCC(=O)O", "achiral"),                  # Gly
    ("CC(C)(N)C(=O)O", "achiral"),            # Aib
    ("CC[C@H](C)[C@H](N)C(=O)O", "L"),        # L-Ile
    ("CC[C@H](C)[C@@H](N)C(=O)O", "D"),       # D-allo-Ile: alpha flipped, beta kept
    ("CC[C@@H](C)[C@H](N)C(=O)O", "L"),       # L-allo-Ile
    ("NCCC[C@H](N)C(=O)O", "L"),              # L-Orn
    ("NCCC[C@@H](N)C(=O)O", "D"),             # D-Orn
])
def test_corn_stereo(smiles, expected):
    assert _stereo(smiles) == expected


def test_unspecified_stereocenter_is_undetermined():
    assert _stereo("CC(N)C(=O)O") == "undetermined"


def test_template_match_standard_residues():
    assert _standard_hit("C[C@H](N)C(=O)O") == {"A": "L"}
    assert _standard_hit("C[C@@H](N)C(=O)O") == {"A": "D"}
    assert _standard_hit("CC[C@H](C)[C@H](N)C(=O)O") == {"I": "L"}
    assert _standard_hit("CC[C@H](C)[C@@H](N)C(=O)O") == {"I": "D"}   # D-allo-Ile matches the mirror of Ile
    assert _standard_hit("CC[C@@H](C)[C@H](N)C(=O)O") == {}           # allo is not a standard residue
    assert _standard_hit("NCC(=O)O") == {"G": "achiral"}
    assert _standard_hit("CC(C)(N)C(=O)O") == {}                      # Aib is not standard
    assert _standard_hit("NCCC[C@H](N)C(=O)O") == {}                  # Orn is not standard


def test_ala_does_not_match_ser_or_cys():
    assert _standard_hit("OC[C@H](N)C(=O)O") == {"S": "L"}
    assert _standard_hit("SC[C@H](N)C(=O)O") == {"C": "L"}


def test_in_chain_stereo_and_order():
    # L-Ala-D-Ala-Gly (free termini)
    mol = Chem.MolFromSmiles("C[C@H](N)C(=O)N[C@H](C)C(=O)NCC(=O)O")
    order, topo = chain_order(mol, alpha_sites(mol))
    assert topo == "linear"
    assert [corn_stereo(mol, s)[0] for s in order] == ["L", "D", "achiral"]
