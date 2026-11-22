"""
Figures and numbers for the Murcko-scaffold clustering deck, straight from the saved data so
the slides cannot drift from the notebook:

  reports/slide_figs/m1_organisms.png         peptides per organism + organisms tested per peptide
  reports/slide_figs/m2_smiles_peptide.png    one short real peptide drawn from its SMILES
  reports/slide_figs/m3_tautomers.png         the His HID / HIE variants of one peptide
  reports/slide_figs/m4_murcko_step.png       peptide -> Murcko scaffold (scaffold atoms highlighted)
  reports/slide_figs/m5_cluster_sizes.png     cluster-size distribution
  reports/slide_figs/m6_top_scaffolds.png     the five largest scaffolds
  reports/slide_figs/m_deck_facts.json        every number the deck quotes

Inputs: data/dbaasp_dev_full.parquet, data/dbaasp_dev_3org_murcko_clusters.parquet,
        .cache/dbaasp_dev_raw.jsonl (tautomer descriptions).

  python scripts/slides/build_murcko_deck_figs.py
"""
import io
import json
import re
from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "slide_figs"
OUT.mkdir(exist_ok=True)

BG, DARK, TEAL, ORANGE, GRAY = "#F6F5F1", "#16212E", "#0F7B6C", "#D9822B", "#6B7280"
THREE = ["Escherichia coli", "Staphylococcus aureus", "Pseudomonas aeruginosa"]

full = pd.read_parquet(REPO / "data" / "dbaasp_dev_full.parquet")
clu = pd.read_parquet(REPO / "data" / "dbaasp_dev_3org_murcko_clusters.parquet")
raw = {}
with open(REPO / ".cache" / "dbaasp_dev_raw.jsonl") as f:
    for line in f:
        r = json.loads(line)
        raw[r["id"]] = r

facts = {}

# ---------------------------------------------------------------- data (slide 3)
rows3 = full[full["organism"].isin(THREE)]
peps3 = rows3.drop_duplicates("id")
n_org_per_pep = rows3.groupby("id")["organism"].nunique()
facts["all_peptides"] = int(full["id"].nunique())
facts["united_rows"] = int(len(full))
facts["activity_rows_all"] = int(full["activity_id"].notna().sum())
facts["rows_3org"] = int(len(rows3))
facts["peptides_3org"] = int(peps3["id"].nunique())
facts["per_org"] = {o: int(rows3[rows3.organism == o]["id"].nunique()) for o in THREE}
facts["orgs_per_peptide"] = {int(k): int(v) for k, v in n_org_per_pep.value_counts().sort_index().items()}
facts["raw_strain_names"] = int(full["target_species"].nunique())
facts["organisms_after_merge"] = int(full["organism"].nunique())

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": BG, "figure.facecolor": BG,
                     "text.color": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK})
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
short = {"Escherichia coli": "E. coli", "Staphylococcus aureus": "S. aureus", "Pseudomonas aeruginosa": "P. aeruginosa"}
vals = [facts["per_org"][o] for o in THREE]
b = ax[0].bar([short[o] for o in THREE], vals, color=[TEAL, ORANGE, "#3D5A80"], width=0.6)
for r, v in zip(b, vals):
    ax[0].text(r.get_x() + r.get_width() / 2, v + 250, f"{v:,}", ha="center", fontsize=15, fontweight="bold")
ax[0].set_title("Peptides with data per organism", fontsize=16, loc="left", fontweight="bold")
ax[0].set_ylim(0, max(vals) * 1.15)
vc = n_org_per_pep.value_counts().sort_index()
b2 = ax[1].bar([f"{k} of 3" for k in vc.index], vc.values, color=GRAY, width=0.6)
for r, v in zip(b2, vc.values):
    ax[1].text(r.get_x() + r.get_width() / 2, v + 250, f"{v:,}", ha="center", fontsize=15, fontweight="bold")
ax[1].set_title("Organisms tested per peptide", fontsize=16, loc="left", fontweight="bold")
ax[1].set_ylim(0, vc.max() * 1.15)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=14)
plt.tight_layout()
fig.savefig(OUT / "m1_organisms.png", dpi=160)
plt.close(fig)


def png_from_drawer(d, path):
    d.FinishDrawing()
    Image.open(io.BytesIO(d.GetDrawingText())).save(path)


def draw_mol(mol, path, size=(1000, 560), highlight=None, hl_color=(0.85, 0.51, 0.17)):
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions()
    o.setBackgroundColour((0.965, 0.961, 0.945, 1))
    o.bondLineWidth = 2
    if highlight:
        d.DrawMolecule(mol, highlightAtoms=list(highlight), highlightAtomColors={a: hl_color for a in highlight})
    else:
        d.DrawMolecule(mol)
    png_from_drawer(d, path)


# ------------------------------------------------- SMILES (slide 4) and tautomers (slide 5)
def desc(s):
    return re.sub(r"\s+", " ", s.get("description") or "")


# slide 5: shortest peptide whose only ambiguity is His (two SMILES, HID + HIE)
cands = []
for pid, r in raw.items():
    sm = [s for s in r.get("smiles") or [] if s.get("smiles")]
    if len(sm) == 2 and all("Histidine:" in desc(s) and "Arginine" not in desc(s) for s in sm):
        seq = r.get("sequence") or ""
        if 3 <= len(seq) <= 6 and "H" in seq and set(seq) <= set("ACDEFGHIKLMNPQRSTVWY") and not r.get("nTerminus") and not r.get("cTerminus"):
            m = Chem.MolFromSmiles(sm[0]["smiles"])
            # a clean linear peptide has exactly the rings its side chains bring (H, F, Y, P: 1 each; W: 2)
            expected_rings = sum(seq.count(a) for a in "HFYP") + 2 * seq.count("W")
            if m is not None and m.GetRingInfo().NumRings() == expected_rings:
                cands.append((len(seq), pid))
cands.sort()
his_pid = cands[0][1]
his = raw[his_pid]
hid = next(s for s in his["smiles"] if "HID" in desc(s))["smiles"]
hie = next(s for s in his["smiles"] if "HIE" in desc(s))["smiles"]
facts["his_example"] = {"id": his_pid, "name": his.get("name"), "sequence": his.get("sequence")}
facts["his_example"]["same_canonical"] = Chem.MolToSmiles(Chem.MolFromSmiles(hid)) == Chem.MolToSmiles(Chem.MolFromSmiles(hie))
ms = [Chem.MolFromSmiles(hid), Chem.MolFromSmiles(hie)]
img = Draw.MolsToGridImage(ms, molsPerRow=2, subImgSize=(620, 440), legends=["HID  (N-H on one ring nitrogen)", "HIE  (N-H on the other)"])
img.save(OUT / "m3_tautomers.png")

# slide 4: the same short peptide as a single drawing
draw_mol(ms[0], OUT / "m2_smiles_peptide.png", size=(1100, 600))
facts["smiles_example"] = {"id": his_pid, "sequence": his.get("sequence"), "smiles": hid,
                           "heavy_atoms": ms[0].GetNumHeavyAtoms()}

# ------------------------------------------------------------------ Murcko step (slide 7)
def has(seq, *aa):
    return all(a in seq for a in aa)


kept = clu[clu["sequence_full"].str.len().between(4, 7) & clu["sequence_full"].map(lambda s: has(s, "W", "F"))
           & (clu["murcko_scaffold"] != "no_scaffold") & ~clu["sequence_full"].str.contains("[a-zX/]", regex=True)]
ex = kept.sort_values("sequenceLength").iloc[0]
mol = Chem.MolFromSmiles(ex["smiles_fixed"])
scaf = Chem.MolFromSmiles(ex["murcko_scaffold"])
hl = mol.GetSubstructMatch(scaf)
draw_mol(mol, OUT / "m4_murcko_step.png", size=(1250, 620), highlight=hl)
draw_mol(scaf, OUT / "m4_murcko_scaffold.png", size=(900, 520))
facts["murcko_example"] = {"id": int(ex["id"]), "name": ex["name"], "sequence": ex["sequence_full"],
                           "scaffold": ex["murcko_scaffold"], "cluster_size": int(ex["murcko_cluster_size"])}

# ---------------------------------------------------------------- cluster numbers (slides 6, 8-10)
facts["peptides_with_smiles_3org"] = int(clu["id"].nunique() + sum(len(x) for x in clu["duplicate_ids"]))
facts["peptides_no_smiles"] = facts["peptides_3org"] - facts["peptides_with_smiles_3org"]
facts["peptides_clustered"] = int(len(clu))
facts["duplicate_groups"] = int(clu["duplicate_ids"].map(len).gt(0).sum())
facts["duplicate_ids_removed"] = int(clu["duplicate_ids"].map(len).sum())
facts["variant_rule"] = clu["variant_rule"].value_counts().to_dict()
facts["n_variants"] = {int(k): int(v) for k, v in clu["n_variants"].value_counts().sort_index().items()}

cs = clu.groupby("murcko_cluster_id").agg(n=("id", "size"), scaf=("murcko_scaffold", "first")).sort_values("n", ascending=False)
facts["n_clusters"] = int(len(cs))
facts["singletons"] = int((cs.n == 1).sum())
facts["no_scaffold"] = int(cs[cs.scaf == "no_scaffold"].n.sum())
buckets = pd.cut(cs.n, [0, 1, 2, 5, 10, 50, 10**7], labels=["1", "2", "3-5", "6-10", "11-50", ">50"])
bc = cs.groupby(buckets, observed=False).n.agg(["size", "sum"])
facts["buckets"] = {k: {"clusters": int(v["size"]), "peptides": int(v["sum"])} for k, v in bc.iterrows()}
facts["peptides_in_clusters_ge6"] = int(cs[cs.n >= 6].n.sum())
facts["share_in_clusters_ge6"] = round(facts["peptides_in_clusters_ge6"] / len(clu), 3)
facts["share_singleton_clusters"] = round(facts["singletons"] / len(cs), 3)
facts["top5"] = [{"scaffold": r.scaf, "n": int(r.n)} for r in cs.head(6).itertuples()]

# ring-systems-only variant (notebook section 5e): linker atoms between rings dropped
rs = clu["ringsys_key"].value_counts()
rs_ringed = rs.drop("no_scaffold", errors="ignore")
multi = clu[clu["ringsys_key"].str.contains("|", regex=False)]
facts["ringsys"] = {
    "clusters": int(len(rs)), "singletons": int((rs == 1).sum()),
    "ring_free": int(rs.get("no_scaffold", 0)), "largest_ring_bearing": int(rs_ringed.max()),
    "multi_ring_peptides": int(len(multi)),
    "multi_distinct_scaffolds": int(multi["murcko_scaffold"].nunique()),
    "multi_distinct_ringsys": int(multi["ringsys_key"].nunique()),
    "murcko_largest_ring_bearing": int(cs[cs.scaf != "no_scaffold"].n.max()),
}

# one ring combination, several linkers: the most common multi-ring key and its three biggest plain scaffolds
multi_key = rs[rs.index.str.contains("|", regex=False)].index[0]
linker_scafs = clu[clu["ringsys_key"] == multi_key]["murcko_scaffold"].value_counts().head(3)
facts["linker_example"] = {"ringsys_key": multi_key, "n_peptides": int(rs[multi_key]),
                           "plain_scaffolds": int(clu[clu["ringsys_key"] == multi_key]["murcko_scaffold"].nunique()),
                           "top": {k: int(v) for k, v in linker_scafs.items()}}
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in linker_scafs.index], molsPerRow=1, subImgSize=(1200, 220),
                     legends=[f"{n:,} peptides" for n in linker_scafs.values]).save(OUT / "m7_linker_example.png")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
ax[0].bar(bc.index.astype(str), bc["size"], color=TEAL, width=0.65)
for i, v in enumerate(bc["size"]):
    ax[0].text(i, v + bc["size"].max() * 0.015, f"{v:,}", ha="center", fontsize=14, fontweight="bold")
ax[0].set_title("Clusters by cluster size", fontsize=16, loc="left", fontweight="bold")
ax[0].set_xlabel("peptides in the cluster", fontsize=13)
ax[1].bar(bc.index.astype(str), bc["sum"], color=ORANGE, width=0.65)
for i, v in enumerate(bc["sum"]):
    ax[1].text(i, v + bc["sum"].max() * 0.015, f"{v:,}", ha="center", fontsize=14, fontweight="bold")
ax[1].set_title("Peptides by size of their cluster", fontsize=16, loc="left", fontweight="bold")
ax[1].set_xlabel("peptides in the cluster", fontsize=13)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=13)
    a.set_yticks([])
plt.tight_layout()
fig.savefig(OUT / "m5_cluster_sizes.png", dpi=160)
plt.close(fig)

top = [r for r in cs.head(5).itertuples()]
mols = [Chem.MolFromSmiles(r.scaf) if r.scaf != "no_scaffold" else Chem.MolFromSmiles("*") for r in top]
leg = [f"{'no ring' if r.scaf == 'no_scaffold' else r.scaf}  -  {r.n:,} peptides" for r in top]
Draw.MolsToGridImage(mols[1:], molsPerRow=2, subImgSize=(430, 330), legends=leg[1:]).save(OUT / "m6_top_scaffolds.png")

# example cluster (slide 10): the proline cluster
ex_scaf = "C1CCNC1"
ec = clu[clu["murcko_scaffold"] == ex_scaf]
facts["example_cluster"] = {
    "scaffold": ex_scaf, "n": int(len(ec)),
    "length_min": int(ec["sequenceLength"].min()), "length_median": float(ec["sequenceLength"].median()),
    "length_max": int(ec["sequenceLength"].max()),
    "organisms": {o: int(ec["organisms"].map(lambda x: o in list(x)).sum()) for o in THREE},
    "samples": [{"id": int(r.id), "name": (r.name or "")[:34], "sequence": r.sequence_full[:34], "length": int(r.sequenceLength)}
                for r in ec.sort_values("sequenceLength").iloc[[0, len(ec) // 4, len(ec) // 2, 3 * len(ec) // 4, len(ec) - 1]].itertuples()],
}
dups = clu[clu["duplicate_ids"].map(len) > 0].sort_values("duplicate_ids", key=lambda s: s.map(len), ascending=False).head(4)
facts["dup_examples"] = [{"ids": [int(r.id)] + [int(x) for x in r.duplicate_ids], "name": (r.name or "")[:30],
                          "sequence": r.sequence_full[:24]} for r in dups.itertuples()]
facts["dup_groups_largest"] = int(clu["duplicate_ids"].map(len).max() + 1)

(OUT / "m_deck_facts.json").write_text(json.dumps(facts, indent=1, default=str))
print(json.dumps(facts, indent=1, default=str))
