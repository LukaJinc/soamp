"""Alpha-carbon stereochemistry from a SMILES, without CIP labels.

Why not R/S: L-Cys is R while almost every other L-residue is S, and a
side-chain change can flip the label. Two label-free methods are used and
reported separately:

1. ``corn_stereo`` -- local handedness of the alpha carbon expressed in
   backbone roles (looking from H: CO -> R -> N clockwise = L, the CORN rule).
   It needs no knowledge of the side chain, so it works for any residue.
2. ``match_template`` -- chirality-aware substructure match (RDKit
   ``useChirality=True``) of the residue against a reference template (the
   20 standard L-residues, and DBAASP's own reference SMILES for unusual
   residues). The mirror-image template is tried too; a hit on the mirror
   means the opposite configuration of the template.

Which alpha carbons are achiral (Gly, Aib, N-substituted glycines) comes from
RDKit's stereo perception (``FindPotentialStereo``), not from a name list.
"""
from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

CW = Chem.ChiralType.CHI_TETRAHEDRAL_CW
CCW = Chem.ChiralType.CHI_TETRAHEDRAL_CCW

# N - C(alpha, sp3) - C(=O): matches peptide-bonded and free termini, N-methyl,
# proline and peptoid (N-substituted glycine) residues alike.
_BACKBONE = Chem.MolFromSmarts("[#7;!$([#7]=*)]-[CX4]-[CX3]=[OX1]")
_CARBOXYL_OH = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")

STANDARD_L = {  # free L-amino acids, PubChem isomeric SMILES
    "A": "C[C@H](N)C(=O)O", "R": "N=C(N)NCCC[C@H](N)C(=O)O", "N": "NC(=O)C[C@H](N)C(=O)O",
    "D": "OC(=O)C[C@H](N)C(=O)O", "C": "SC[C@H](N)C(=O)O", "Q": "NC(=O)CC[C@H](N)C(=O)O",
    "E": "OC(=O)CC[C@H](N)C(=O)O", "G": "NCC(=O)O", "H": "N[C@@H](Cc1cnc[nH]1)C(=O)O",
    "I": "CC[C@H](C)[C@H](N)C(=O)O", "L": "CC(C)C[C@H](N)C(=O)O", "K": "NCCCC[C@H](N)C(=O)O",
    "M": "CSCC[C@H](N)C(=O)O", "F": "c1ccccc1C[C@H](N)C(=O)O", "P": "OC(=O)[C@@H]1CCCN1",
    "S": "OC[C@H](N)C(=O)O", "T": "C[C@@H](O)[C@H](N)C(=O)O",
    "W": "c1ccc2c(c1)c(c[nH]2)C[C@H](N)C(=O)O", "Y": "Oc1ccc(cc1)C[C@H](N)C(=O)O",
    "V": "CC(C)[C@H](N)C(=O)O",
}


@dataclass(frozen=True)
class AlphaSite:
    alpha: int   # alpha-carbon atom index
    n: int       # backbone N
    co: int      # backbone carbonyl C


def alpha_sites(mol: Chem.Mol) -> list[AlphaSite]:
    """One site per alpha carbon (deduplicated on the carbon)."""
    seen: dict[int, AlphaSite] = {}
    for n, ca, c, _o in mol.GetSubstructMatches(_BACKBONE):
        seen.setdefault(ca, AlphaSite(ca, n, c))
    return list(seen.values())


def chain_order(mol: Chem.Mol, sites: list[AlphaSite]) -> tuple[list[AlphaSite] | None, str]:
    """Sites in N->C order and a topology tag: linear | cyclic | branched.

    Residue j follows residue i when j's backbone N is bonded to i's carbonyl C.
    Cyclic (head-to-tail) and multi-chain molecules have no unique start, so no
    order is returned for them.
    """
    by_n = {s.n: s for s in sites}
    nxt: dict[int, AlphaSite] = {}
    for s in sites:
        for nb in mol.GetAtomWithIdx(s.co).GetNeighbors():
            if nb.GetIdx() in by_n and nb.GetIdx() != s.n:
                nxt[s.alpha] = by_n[nb.GetIdx()]
    followers = {t.alpha for t in nxt.values()}
    starts = [s for s in sites if s.alpha not in followers]
    if len(starts) == 1:
        order, cur = [], starts[0]
        while cur is not None and len(order) <= len(sites):
            order.append(cur)
            cur = nxt.get(cur.alpha)
        if len(order) == len(sites):
            return order, "linear"
    if not starts and len(nxt) == len(sites):
        return None, "cyclic"
    return None, "branched"


def _permutation_is_odd(seq: list[int], reference: list[int]) -> bool:
    pos = [reference.index(x) for x in seq]
    inversions = sum(pos[i] > pos[j] for i in range(len(pos)) for j in range(i + 1, len(pos)))
    return inversions % 2 == 1


def _backbone_handedness(molh: Chem.Mol, site: AlphaSite) -> str | None:
    """'CW'/'CCW' of (H, CO, R, N) around the alpha carbon, or None."""
    atom = molh.GetAtomWithIdx(site.alpha)
    if atom.GetChiralTag() not in (CW, CCW):
        return None
    nbrs = [b.GetOtherAtomIdx(site.alpha) for b in atom.GetBonds()]
    hs = [i for i in nbrs if molh.GetAtomWithIdx(i).GetAtomicNum() == 1]
    rest = [i for i in nbrs if i not in (site.n, site.co) and i not in hs]
    if len(nbrs) != 4 or len(hs) != 1 or len(rest) != 1:
        return None  # quaternary alpha carbon or odd valence: CORN does not apply
    tag = atom.GetChiralTag()
    if _permutation_is_odd(nbrs, [hs[0], site.co, rest[0], site.n]):
        tag = CCW if tag == CW else CW
    return "CW" if tag == CW else "CCW"


@lru_cache(maxsize=1)
def _l_reference_handedness() -> str:
    """Handedness of L-alanine, computed with this same code path."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
    site = alpha_sites(mol)[0]
    return _backbone_handedness(mol, site)


def corn_stereo(mol: Chem.Mol, site: AlphaSite) -> tuple[str, str]:
    """(stereo, reason): stereo is L | D | achiral | undetermined."""
    info = [s for s in Chem.FindPotentialStereo(mol)
            if s.type == Chem.StereoType.Atom_Tetrahedral and s.centeredOn == site.alpha]
    if not info:
        return "achiral", ""
    if info[0].specified != Chem.StereoSpecified.Specified:
        return "undetermined", "alpha carbon is a stereocenter but its chirality is unspecified"
    hand = _backbone_handedness(Chem.AddHs(mol), site)
    if hand is None:
        return "undetermined", "alpha carbon has no single H (quaternary or unusual valence)"
    return ("L" if hand == _l_reference_handedness() else "D"), ""


# ---- template matching ----------------------------------------------------

@dataclass(frozen=True)
class Template:
    name: str
    query: Chem.Mol
    alpha: int   # alpha carbon index in the query
    co: int      # carbonyl carbon index in the query (may gain neighbors in a chain)
    stereo: str  # configuration of the template itself: L | D | achiral


def build_template(name: str, smiles: str) -> Template | None:
    """Reference residue -> peptide-context query (terminal carboxyl OH removed)."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    sites = alpha_sites(mol)
    if len(sites) != 1:
        return None
    stereo, _ = corn_stereo(mol, sites[0])
    rw = Chem.RWMol(mol)
    hit = rw.GetSubstructMatch(_CARBOXYL_OH)
    if hit:
        rw.RemoveAtom(hit[2])
    query = rw.GetMol()
    query.UpdatePropertyCache(strict=False)
    qsites = alpha_sites(query)
    if len(qsites) != 1:
        return None
    return Template(name, query, qsites[0].alpha, qsites[0].co, stereo)


def _mirror(query: Chem.Mol, alpha: int) -> Chem.Mol:
    mirror = Chem.Mol(query)
    atom = mirror.GetAtomWithIdx(alpha)
    atom.SetChiralTag(CCW if atom.GetChiralTag() == CW else CW)
    return mirror


def _side_chain_fits(mol: Chem.Mol, template: Template, match: tuple[int, ...]) -> bool:
    """Side-chain carbons must have exactly the template's heavy-atom degree (so
    Ala does not match Ser); heteroatoms may carry extra substituents (disulfide
    S, acylated Lys N), and the carbonyl C continues into the next residue."""
    for q_idx, t_idx in enumerate(match):
        q_atom = template.query.GetAtomWithIdx(q_idx)
        if q_atom.GetAtomicNum() == 6 and q_idx != template.co:
            if mol.GetAtomWithIdx(t_idx).GetDegree() != q_atom.GetDegree():
                return False
    return True


def match_template(mol: Chem.Mol, site: AlphaSite, template: Template) -> str | None:
    """'same' / 'mirror' relative to the template, or None if the residue is
    not this template (side chain differs)."""
    for label, query in (("same", template.query), ("mirror", _mirror(template.query, template.alpha))):
        for match in mol.GetSubstructMatches(query, useChirality=True, uniquify=False, maxMatches=2000):
            if match[template.alpha] == site.alpha and _side_chain_fits(mol, template, match):
                return label
    return None


def template_stereo(template: Template, relation: str) -> str:
    if template.stereo == "achiral":
        return "achiral"
    if template.stereo not in ("L", "D"):
        return "undetermined"
    if relation == "same":
        return template.stereo
    return "D" if template.stereo == "L" else "L"


@lru_cache(maxsize=1)
def standard_templates() -> dict[str, Template]:
    out = {}
    for letter, smi in STANDARD_L.items():
        tpl = build_template(letter, smi)
        if tpl is not None:
            out[letter] = tpl
    return out
