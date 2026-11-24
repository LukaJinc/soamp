"""
Stereo audit step 4: database-level counts, twin results and the D->L alignment
sensitivity check, written to <outputs.dir>/summary.md.

The sensitivity check recomputes the splitting graph's sequence-identity edges
in memory (same aligner and parameters as soamp.splitting.alignment) with the
current tokens and with D-residues mapped to L, and counts what changes. No
split, config or data file is modified.
"""
import argparse
import json

import igraph as ig
import numpy as np
import pandas as pd

from soamp.splitting.alignment import TokenAlphabet, compute_edges
from soamp.splitting.tokens import MOD_PREFIX, tokenize_linear
from soamp.stereo.config import StereoConfig
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/stereo/base.yaml")
    return p.parse_args()


def md(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids a tabulate dependency)."""
    if not len(df):
        return "_none_"
    head = [str(df.index.name or "")] + [str(c) for c in df.columns]
    rows = [[str(i)] + [f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in r]
            for i, r in zip(df.index, df.itertuples(index=False))]
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def d_to_l(tokens: list[str]) -> list[str]:
    return [MOD_PREFIX + t[len(MOD_PREFIX) + 2:] if t.startswith(MOD_PREFIX + "D-") else t for t in tokens]


def alignment_sensitivity(cfg, log) -> pd.DataFrame:
    processed = pd.read_parquet(cfg.inputs.processed_parquet, columns=["peptide_id", "is_linear"]).drop_duplicates("peptide_id")
    linear = set(processed.loc[processed["is_linear"], "peptide_id"])
    tokens = {}
    with open(cfg.inputs.raw_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in linear and r.get("sequence"):
                t = tokenize_linear(r["sequence"], r.get("unusualAminoAcids"))
                if t:
                    tokens[r["id"]] = t
    lo = min(cfg.alignment.identity_thresholds)
    results = {}
    for variant, fn in (("current_tokens", lambda t: t), ("d_mapped_to_l", d_to_l)):
        seqs = sorted({tuple(fn(t)) for t in tokens.values()})
        alphabet = TokenAlphabet(sorted({x for s in seqs for x in s if x.startswith(MOD_PREFIX)}))
        enc = [alphabet.encode(s) for s in seqs]
        edges = compute_edges(enc, alphabet, lo, cfg.alignment.gap_open, cfg.alignment.gap_extend, cfg.alignment.n_jobs)
        results[variant] = (seqs, edges)
        log.info("%s: %d distinct sequences, %d edges >= %.2f", variant, len(seqs), len(edges), lo)
    # compare on peptides: collapse each variant's edges to peptide-id pairs
    pep_edges = {}
    for variant, (seqs, edges) in results.items():
        idx = {s: i for i, s in enumerate(seqs)}
        fn = (lambda t: t) if variant == "current_tokens" else d_to_l
        node = {pid: idx[tuple(fn(t))] for pid, t in tokens.items()}
        by_node: dict[int, list[int]] = {}
        for pid, n in node.items():
            by_node.setdefault(n, []).append(pid)
        pep_edges[variant] = (node, edges, by_node)
    rows = []
    for thr in cfg.alignment.identity_thresholds:
        row = {"identity_threshold": thr}
        for variant, (node, edges, by_node) in pep_edges.items():
            kept = [e for e, v in edges.items() if v >= thr]
            g = ig.Graph(n=len(by_node and max(by_node) + 1), edges=kept)
            comp = g.connected_components().membership
            sizes = pd.Series([comp[n] for n in node.values()]).value_counts()
            row[f"{variant}: sequence pairs"] = len(kept)
            row[f"{variant}: components"] = int(len(sizes))
            row[f"{variant}: largest component (peptides)"] = int(sizes.iloc[0])
        cur, mapped = pep_edges["current_tokens"], pep_edges["d_mapped_to_l"]
        merged_nodes = sum(len(v) for v in mapped[2].values() if len({cur[0][p] for p in v}) > 1)
        row["peptides in a group that D->L merges (identical after mapping)"] = merged_nodes
        rows.append(row)
    return pd.DataFrame(rows).set_index("identity_threshold").T


def main() -> None:
    log = configure_logging("stereo.04_summary")
    cfg = load_config(_parse_args().config, StereoConfig)
    out = cfg.outputs.dir
    pep = pd.read_csv(out / cfg.outputs.peptides)
    res = pd.read_parquet(out / "residues_annotated.parquet")
    twins = pd.read_csv(out / cfg.outputs.twins)
    mic = pd.read_csv(out / cfg.outputs.twin_mic) if (out / cfg.outputs.twin_mic).exists() else pd.DataFrame()
    lines = [f"# L/D stereochemistry audit (dev corpus, {len(pep)} peptides)\n"]
    pep["kind"] = np.where(pep["is_linear"], "linear", "non-linear")

    lines += ["## Peptides by label (notation)\n", md(pd.crosstab(pep["label_notation"], pep["kind"], margins=True)), ""]
    lines += ["## Peptides by label (SMILES, positional only)\n",
              md(pd.crosstab(pep["label_smiles"].fillna("not_mapped"), pep["kind"], margins=True)), ""]
    lines += ["## SMILES mapping status\n", md(pep["mapping_status"].str.split("(").str[0].value_counts().to_frame("peptides")), ""]
    lines += ["## Agreement notation vs SMILES (peptide level)\n", md(pep["agreement_status"].value_counts().to_frame("peptides")), ""]
    res_pos = res[res["smiles_stereo_corn"].notna()]
    lines += ["## Residue-level agreement (positional peptides)\n", md(res_pos["agreement"].value_counts().to_frame("residues")), ""]
    lines += ["## Residue-level: SMILES backbone-role vs template match\n",
              md(pd.crosstab(res_pos["smiles_stereo_corn"], res_pos["smiles_stereo_template"].fillna("none"))), ""]
    std = res_pos[res_pos["residue_class"] == "standard"]
    lines += [f"Standard residues whose own template matched in SMILES: {(std['template_relation'].notna()).mean():.4f} "
              f"({int(std['template_relation'].notna().sum())}/{len(std)}) -- a check that sequence positions and SMILES residues line up.\n"]

    lines += ["## D-residues by residue class (notation)\n",
              md(pd.crosstab(res["residue_class"], res["notation_stereo"], margins=True)), ""]
    dn = res[(res["residue_class"] == "nonstandard") & (res["notation_stereo"] == "D")]
    lines += ["Non-standard residues that occur in D-form (top 25):\n", md(dn["token"].value_counts().head(25).to_frame("count")), ""]
    d_all = res[res["notation_stereo"] == "D"].copy()
    L = res.groupby("peptide_id")["position"].transform("max")
    d_all["where"] = np.where(d_all["position"].eq(1) | d_all["position"].eq(L.loc[d_all.index]), "terminal", "internal")
    lines += ["D-residue positions (coarse):\n", md(d_all["where"].value_counts().to_frame("D residues")), ""]
    alt = res[res["notation_stereo"].isin(["L", "D"])].groupby("peptide_id")["notation_stereo"].agg(
        lambda s: len(s) >= 4 and all(a != b for a, b in zip(s.iloc[:-1], s.iloc[1:])))
    lines += [f"Peptides with strictly alternating L/D (>=4 chiral residues): {int(alt.sum())}\n"]
    amb = pep[(pep["has_nonstandard"]) & (pep["has_D_nonstandard_notation"])]
    lines += [f"Peptides with >1 kind of ambiguity (non-standard AND D-form non-standard): {len(amb)}\n"]
    lines += ["## Coverage gaps\n",
              md(res["notation_reason"].replace("", np.nan).dropna().str.replace(r"'.'|position \d+", "...", regex=True).value_counts().head(10).to_frame("residues")),
              md(res[res["smiles_reason"] != ""]["smiles_reason"].str.replace(r"\(.*\)", "", regex=True).value_counts().head(10).to_frame("residues")), ""]

    lines += ["## Stereo-only twins\n",
              f"- pairs: {len(twins)} (same annotation: {int(twins['same_annotation'].sum()) if len(twins) else 0}; "
              f"also differing in termini/bonds: {int((~twins['same_annotation']).sum()) if len(twins) else 0})",
              f"- peptides involved: {len(set(twins['peptide_a']) | set(twins['peptide_b'])) if len(twins) else 0}",
              f"- examples: {twins.head(8)[['peptide_a', 'peptide_b']].values.tolist() if len(twins) else '[]'}"]
    if len(twins):
        lines += [f"- pairs in different split/fold buckets: {int((twins['different_bucket'] == True).sum())} "  # noqa: E712
                  f"(same bucket: {int((twins['different_bucket'] == False).sum())}; not in split: {int(twins['different_bucket'].isna().sum())})"]
    if len(mic):
        q = mic["abs_log2_mic_ratio"].quantile([0.25, 0.5, 0.75, 0.95]).round(2).to_dict()
        lines += [f"- pairs with a MIC for the same organism: {mic[['peptide_a', 'peptide_b']].drop_duplicates().shape[0]} "
                  f"({len(mic)} pair-organism rows); |log2 MIC ratio| quantiles {q}; "
                  f"within 1 dilution (<=1): {(mic['abs_log2_mic_ratio'] <= 1).mean():.2f}"]
    lines += [""]

    log.info("alignment sensitivity ...")
    sens = alignment_sensitivity(cfg, log)
    lines += ["## Would mapping D->L change the grouping? (linear, tokenizable peptides)\n", md(sens), ""]
    (out / cfg.outputs.summary).write_text("\n".join(lines))
    log.info("wrote %s", out / cfg.outputs.summary)


if __name__ == "__main__":
    main()
