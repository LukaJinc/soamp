"""
Faithful re-implementation of QMAP's peptide-level inclusion filter (the *first*
five checks in build_dbaasp_dataset(), data/dbaasp/build_dataset.py), used to
explain -- for any DBAASP peptide NOT present in QMAP's published dataset -- which
specific check it would fail, in the same order QMAP's code evaluates them
(the loop uses `continue` on first failure, so order = actual causal reason).

IMPORTANT finding from reading QMAP's source directly (not assumed): the
non-canonical-residue check only excludes a peptide when a "X" placeholder
remains in the common_sequence AND no native SMILES is available:

    if "X" in peptide.common_sequence.upper() and len(peptide.smiles) == 0:
        continue

So a peptide with a non-canonical residue that already has a native DBAASP SMILES
is NOT excluded by QMAP for residue reasons -- it passes this check regardless of
which residue it is. The two unconditional, SMILES-independent exclusion axes are
(1) bond type outside {DSB, AMD}, and (2) N-/C-terminal modification outside
{None, ACT} / {None, AMD}. This module encodes that logic exactly (not an
approximation) so the diff step's bucket assignment is a faithful reconstruction
of QMAP's actual behavior, not a re-guessed filter.
"""
from parse_dbaasp import DBAASPPeptide, has_unresolved_placeholder

ORN_DAB_MODIFICATIONS = {"ORN": "O", "D-ORN": "o", "DAB": "B", "D-DAB": "b"}


def qmap_common_sequence(sequence: str, unusual_aa: list) -> str:
    """Reproduces QMAP's replace_common_noncanonical: substitutes ONLY the four
    ORN/D-ORN/DAB/D-DAB codes; every other unusual residue position keeps whatever
    character DBAASP's raw `sequence` field has there (typically 'X')."""
    chars = list(sequence)
    for pos, name in unusual_aa:
        if name in ORN_DAB_MODIFICATIONS:
            chars[pos - 1] = ORN_DAB_MODIFICATIONS[name]
    return "".join(chars)


def qmap_inclusion_check(peptide: DBAASPPeptide):
    """
    Returns (would_be_included: bool, first_failing_reason: str | None).
    Mirrors build_dbaasp_dataset()'s filtering block exactly, in order.
    """
    if peptide.complexity != "Monomer":
        return False, "non_monomer_complexity"

    if peptide.nterminus is not None and peptide.nterminus != "ACT":
        return False, "unsupported_nterminus"

    if peptide.cterminus is not None and peptide.cterminus != "AMD":
        return False, "unsupported_cterminus"

    common_seq = qmap_common_sequence(peptide.sequence, peptide.unusual_amino_acids)
    if "X" in common_seq.upper() and not peptide.has_native_smiles:
        return False, "unresolved_residue_no_native_smiles"

    bad_bonds = [b for b in peptide.intrachain_bonds
                 if (b.get("type") or {}).get("name") not in ("DSB", "AMD")]
    if bad_bonds:
        return False, "nonstandard_bond_type"

    return True, None
