"""
Flattens a raw DBAASP peptide-card JSON payload (as returned by GET /peptides/{id})
into structured Python values used by the step 2 (raw pull) and later steps.
"""
import re

CANONICAL_L = set("ACDEFGHIKLMNPQRSTVWY")


def safe_name(d):
    return d.get("name") if d else None


def has_lowercase_residue(sequence: str) -> bool:
    return any(ch.isalpha() and ch.islower() for ch in sequence)


def has_unresolved_placeholder(sequence: str) -> bool:
    return "X" in sequence.upper() and any(ch.upper() == "X" for ch in sequence)


class DBAASPPeptide:
    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def id(self):
        return self.raw["id"]

    @property
    def dbaasp_id(self):
        return self.raw.get("dbaaspId")

    @property
    def sequence(self):
        return self.raw.get("sequence")

    @property
    def sequence_length(self):
        return self.raw.get("sequenceLength")

    @property
    def complexity(self):
        return safe_name(self.raw.get("complexity"))

    @property
    def synthesis_type(self):
        return safe_name(self.raw.get("synthesisType"))

    @property
    def nterminus(self):
        return safe_name(self.raw.get("nTerminus"))

    @property
    def cterminus(self):
        return safe_name(self.raw.get("cTerminus"))

    @property
    def target_groups(self):
        return [g["name"] for g in (self.raw.get("targetGroups") or [])]

    @property
    def target_objects(self):
        return [o["name"] for o in (self.raw.get("targetObjects") or [])]

    @property
    def unusual_amino_acids(self):
        """List of (position:int, modification_name:str)."""
        out = []
        for aa in self.raw.get("unusualAminoAcids") or []:
            pos = aa.get("position")
            name = (aa.get("modificationType") or {}).get("name")
            if pos is not None and name is not None:
                out.append((pos, name))
        return out

    @property
    def intrachain_bonds(self):
        """List of raw bond dicts (see bond_map.classify_bond for shape)."""
        return self.raw.get("intrachainBonds") or []

    @property
    def native_smiles_list(self):
        return [s["smiles"] for s in (self.raw.get("smiles") or []) if s.get("smiles")]

    @property
    def has_native_smiles(self):
        return len(self.native_smiles_list) > 0

    @property
    def has_noncanonical(self) -> bool:
        seq = self.sequence or ""
        return (len(self.unusual_amino_acids) > 0
                or has_lowercase_residue(seq)
                or has_unresolved_placeholder(seq))

    @property
    def has_d_residue(self) -> bool:
        seq = self.sequence or ""
        if has_lowercase_residue(seq):
            return True
        return any(name.upper().startswith("D-") for _, name in self.unusual_amino_acids)

    @property
    def bond_types(self):
        return sorted({safe_name(b.get("type")) for b in self.intrachain_bonds if b.get("type")})

    @property
    def target_activities(self):
        """List of raw targetActivity dicts."""
        return self.raw.get("targetActivities") or []
