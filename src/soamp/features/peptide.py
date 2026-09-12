"""Pure RDKit-descriptor computation for a peptide SMILES string.

SMILES-only, no dependence on the `sequence` column -- works uniformly
across canonical and non-canonical/cyclic peptides with zero fallback
logic, unlike AA-letter-based composition/hydrophobicity-scale/
amphiphilicity features which would need an undefined fallback for the
~20% of peptides in this dataset that are non-canonical.
"""
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdmolops

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_NAMES = [
    "MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "FractionCSP3", "RingCount", "NumAromaticRings",
    "HeavyAtomCount", "NumHeteroatoms", "LabuteASA", "FormalCharge",
]

_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "TPSA": Descriptors.TPSA,
    "MolLogP": Descriptors.MolLogP,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "FractionCSP3": Descriptors.FractionCSP3,
    "RingCount": Descriptors.RingCount,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "LabuteASA": Descriptors.LabuteASA,
    "FormalCharge": rdmolops.GetFormalCharge,
}


class PeptideFeatureError(ValueError):
    """Raised when a peptide_id maps to more than one distinct SMILES (a
    source-data invariant broken), or when a SMILES that upstream
    curation/QA already validated as parseable fails to parse here."""


def compute_peptide_descriptors(
    smiles: str, descriptor_names: list[str] = DESCRIPTOR_NAMES
) -> dict[str, float] | None:
    """Fixed-order {name: value} dict for descriptor_names, computed from
    smiles via RDKit. Returns None if smiles is empty or
    Chem.MolFromSmiles returns None (invalid SMILES) -- matches this
    repo's existing convention (curation/units.py), never raises on bad
    input at this level."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {name: _DESCRIPTOR_FUNCS[name](mol) for name in descriptor_names}


def select_unique_peptides(rows: list[dict]) -> list[dict]:
    """Dedups mic_classification_dataset.csv rows (repeated once per
    organism) to one {'peptide_id', 'smiles'} per peptide_id,
    order-preserving on first occurrence. Raises PeptideFeatureError if
    the same peptide_id shows two different smiles strings."""
    seen: dict[str, str] = {}
    out = []
    for row in rows:
        pid, smiles = row["peptide_id"], row["smiles"]
        if pid in seen:
            if seen[pid] != smiles:
                raise PeptideFeatureError(
                    f"peptide_id {pid!r} has conflicting smiles values: "
                    f"{seen[pid]!r} vs {smiles!r}"
                )
            continue
        seen[pid] = smiles
        out.append({"peptide_id": pid, "smiles": smiles})
    return out


def build_peptide_feature_rows(
    unique_peptides: list[dict], descriptor_names: list[str] = DESCRIPTOR_NAMES
) -> list[dict]:
    """Top-level composition: compute_peptide_descriptors per unique
    peptide. Raises PeptideFeatureError (fail loud) listing peptide_ids
    whose SMILES didn't parse -- upstream QA already validated
    parseability, so a failure here means an invariant broke, not
    something to silently skip."""
    out = []
    failed = []
    for p in unique_peptides:
        descriptors = compute_peptide_descriptors(p["smiles"], descriptor_names)
        if descriptors is None:
            failed.append(p["peptide_id"])
            continue
        out.append({"peptide_id": p["peptide_id"], **descriptors})
    if failed:
        raise PeptideFeatureError(
            f"{len(failed)} peptide_id(s) had unparseable SMILES: {failed[:10]}"
        )
    return out


def index_feature_rows_by_peptide_id(
    feature_rows: list[dict], descriptor_names: list[str] = DESCRIPTOR_NAMES
) -> dict[str, dict[str, float]]:
    """Reshapes build_peptide_feature_rows-style rows (one dict per peptide,
    string-valued when read back from a CSV) into {peptide_id: {descriptor:
    float}} for PeptideOrganismDataset. Raises PeptideFeatureError if a row
    is missing one of descriptor_names."""
    out = {}
    for row in feature_rows:
        try:
            out[row["peptide_id"]] = {name: float(row[name]) for name in descriptor_names}
        except KeyError as e:
            raise PeptideFeatureError(
                f"peptide_id {row.get('peptide_id')!r} is missing descriptor {e}"
            ) from e
    return out
