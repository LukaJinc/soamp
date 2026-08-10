"""
Classifies DBAASP intrachainBonds against what p2smi's cyclization-constraint system
(p2smi.utilities.smilesgen: SS / HT / SCNT / SCCT / SCSC patterns) can actually build.

p2smi only implements four bond chemistries (verified by reading smilesgen.py's
`constraint_functions` dict and `constrained_peptide_smiles`):
  - SS   : disulfide (S-S) between two Cys-like (disulphide-capable) residues
  - HT   : head-to-tail backbone amide bond (N-terminus amine to C-terminus acid)
  - SCNT : side chain (amine-donor-capable residue) to N-terminus
  - SCCT : side chain (acid-acceptor-capable residue) to C-terminus
  - SCSC : side chain-to-side chain amide/lactam bridge (one amine-donor + one
           acid-acceptor residue, at arbitrary positions -- p2smi's 'N'/'Z' mask
           codes denote chemical role, not literal peptide termini)

DBAASP's bond vocabulary is richer: disulfide (Cystine), backbone amide cyclization
(N-C-termini bond, Diketopiperazine), lactam bridges (Amide bond in the cycle), and
several bond chemistries p2smi has no generator for at all: thioether crosslinks
(Lanthionine, Methyl lanthionine, Sactionine), and lactone/ester bridges
(Cyclic ester). We do not attempt to hand-build SMILES for chemistries p2smi
can't generate -- that risks silently producing an incorrect structure, which is
worse than excluding the entry. Those are reported as unconvertible.
"""

# DBAASP bond.type.name + bond.cycleType.name combinations that map to a p2smi
# constraint we can build with confidence.
SS_CYCLETYPES = {"CST"}          # Cystine (disulfide)
HT_CYCLETYPES = {"NCB", "DKP"}   # N-C-termini bond; 2,5-Diketopiperazine (same amide
                                  # backbone-cyclization chemistry for a 2-residue ring)
# Lactam / amide bridge between side chains, or side chain <-> terminus.
LACTAM_CYCLETYPES = {"LAC"}      # "Amide bond in the cycle (Lactam)"

# Chemistries p2smi has no generator for -- always unconvertible regardless of position.
UNSUPPORTED_CYCLETYPES = {
    "LAN": "Lanthionine (thioether crosslink) -- p2smi has no thioether bond generator",
    "MeLAN": "Methyl lanthionine (thioether crosslink) -- p2smi has no thioether bond generator",
    "SCT": "Sactionine (S-C-alpha thioether linkage) -- p2smi has no thioether bond generator",
    "LTN": "Cyclic ester / lactone -- p2smi's constraint system has no ester-bond generator",
}


def classify_bond(bond: dict):
    """
    bond: one element of DBAASP's intrachainBonds list (raw API shape), i.e.
      {"position1": int, "position2": int, "type": {"name": ...},
       "chainParticipating": {"name": "MMB"|"SSB"|"SMB"}, "cycleType": {"name": ..., "description": ...}}

    Returns dict:
      {"p2smi_constraint": "SS"|"HT"|"SCSC"|"SCNT"|"SCCT"|None,
       "convertible": bool, "reason": str}
    """
    cycletype = (bond.get("cycleType") or {}).get("name")
    chain_part = (bond.get("chainParticipating") or {}).get("name")
    bond_type = (bond.get("type") or {}).get("name")

    if cycletype in SS_CYCLETYPES:
        return {"p2smi_constraint": "SS", "convertible": True, "reason": ""}

    if cycletype in HT_CYCLETYPES:
        return {"p2smi_constraint": "HT", "convertible": True, "reason": ""}

    if cycletype in LACTAM_CYCLETYPES:
        if chain_part == "SSB":
            return {"p2smi_constraint": "SCSC", "convertible": True, "reason": ""}
        elif chain_part == "SMB":
            # side chain <-> mainchain terminus; direction (SCNT vs SCCT) determined by
            # caller using nTerminus/cTerminus + position, since it needs peptide-level
            # context (sequence length) this function doesn't have.
            return {"p2smi_constraint": "SCNT_OR_SCCT", "convertible": True, "reason": ""}
        else:
            return {"p2smi_constraint": None, "convertible": False,
                     "reason": f"lactam bond with unexpected chainParticipating='{chain_part}' "
                               f"(expected SSB or SMB) -- flagged for manual review"}

    if cycletype in UNSUPPORTED_CYCLETYPES:
        return {"p2smi_constraint": None, "convertible": False,
                 "reason": UNSUPPORTED_CYCLETYPES[cycletype]}

    return {"p2smi_constraint": None, "convertible": False,
             "reason": f"unrecognized DBAASP cycleType='{cycletype}' (bond type='{bond_type}') "
                       f"-- no known p2smi mapping, flagged for manual review"}
