"""
Read-only measurement of a saved dev split: every number the methodology deck quotes.

Recomputes the alignment edges with the tokenization of the given config
(``d_residues``, exception table, gaps) and compares them with the saved split:

  * sequence/residue counts and cluster sizes
  * linear sequence pairs per identity band (>=0.9, 0.8-0.9, 0.7-0.8, 0.6-0.7),
    and how many of them sit in different buckets
  * leakage checks: clusters spanning buckets, identical SMILES / identical token
    sequences in more than one bucket, pairs >= hard-link identity across buckets
  * bucket sizes (peptides, activity records, nonlinear share)

Nothing is written except an optional --json file.

  python scripts/splitting/measure_dev_split.py --config config/splitting/dev_v2.yaml \
      --split data/dev/dbaasp_dev_split_v2.parquet [--json out.json]
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from soamp.splitting.alignment import TokenAlphabet, compute_edges, contract_hard_links
from soamp.splitting.dev_config import DevSplitConfig
from soamp.splitting.tokens import build_modification_table, tokenize_linear
from soamp.utils.config import REPO_ROOT, load_config
from soamp.utils.logging import configure_logging

BANDS = [(0.9, 1.01), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7)]
FLOOR = 0.6   # bands are always reported down to 0.6, whatever the config's own threshold


def main() -> None:
    log = configure_logging("splitting.measure_dev_split")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--restrict-to-split-peptides", action="store_true",
                    help="the split covers only part of the config's peptides (e.g. the labelled ones): measure those")
    ap.add_argument("--dataset-csv", default=None,
                    help="classification dataset; records become labelled pairs and active share is reported")
    args = ap.parse_args()
    cfg = load_config(args.config, DevSplitConfig)

    data = pd.read_parquet(REPO_ROOT / "data/dev/dbaasp_dev_processed.parquet")
    data = data[data["target_species"].isin(cfg.organisms)]
    peps = (data.drop_duplicates("peptide_id")[["peptide_id", "sequence_full", "smiles", "is_linear"]]
            .assign(n_rows=lambda d: d["peptide_id"].map(data.groupby("peptide_id").size())))
    split = pd.read_parquet(Path(args.split) if Path(args.split).is_absolute() else REPO_ROOT / args.split)
    if args.restrict_to_split_peptides:
        assert set(split["peptide_id"]) <= set(peps["peptide_id"]), "split has peptides outside the config's organisms"
        peps = peps[peps["peptide_id"].isin(split["peptide_id"])].copy()
    else:
        assert set(split["peptide_id"]) == set(peps["peptide_id"]), "split and config cover different peptides"
    if args.dataset_csv:
        ds = pd.read_csv(Path(args.dataset_csv) if Path(args.dataset_csv).is_absolute() else REPO_ROOT / args.dataset_csv)
        g = ds.groupby("peptide_id")
        assert set(g.groups) == set(peps["peptide_id"]), "dataset and split cover different peptides"
        peps["n_rows"] = peps["peptide_id"].map(g.size())                                   # labelled pairs
        peps["n_active"] = peps["peptide_id"].map(g["label"].apply(lambda s: int((s == "active").sum())))
    peps = peps.merge(split, on="peptide_id")
    peps["bucket"] = peps["split"].where(peps["split"] == "test", "fold_" + peps["fold_id"].astype("Int64").astype(str))

    raw = {}
    with open(REPO_ROOT / ".cache/dbaasp_dev_raw.jsonl") as f:
        for line in f:
            r = json.loads(line)
            raw[r["id"]] = r.get("unusualAminoAcids")
    lin = peps[peps["is_linear"]].copy()
    lin["tokens"] = [tokenize_linear(s, raw[p], cfg.d_residues, cfg.name_exceptions)
                     for p, s in zip(lin["peptide_id"], lin["sequence_full"])]
    scored = lin[lin["tokens"].notna()].copy()
    mods = build_modification_table(scored["tokens"])
    alphabet = TokenAlphabet(list(mods))
    scored["seq"] = scored["tokens"].map(alphabet.encode)
    seqs = sorted(scored["seq"].unique())
    idx = {s: i for i, s in enumerate(seqs)}
    log.info("%d linear peptides tokenized, %d distinct sequences, %d distinct unusual residues", len(scored), len(seqs), len(mods))

    edges = compute_edges(seqs, alphabet, FLOOR, cfg.gap_open, cfg.gap_extension, n_jobs=os.cpu_count() or 1)
    seq_bucket = scored.groupby("seq")["bucket"].agg(lambda s: set(s))
    assert all(len(b) == 1 for b in seq_bucket), "identical token sequences span buckets"
    bucket_of = [next(iter(seq_bucket[s])) for s in seqs]

    out: dict = {"config": args.config, "split": args.split, "d_residues": cfg.d_residues,
                 "identity_threshold": cfg.identity_threshold, "organisms": len(cfg.organisms),
                 "peptides": int(len(peps)), "activity_records": int(peps["n_rows"].sum()),
                 "linear_peptides": int(peps["is_linear"].sum()), "nonlinear_peptides": int((~peps["is_linear"]).sum()),
                 "linear_tokenized": int(len(scored)), "distinct_sequences": len(seqs),
                 "distinct_unusual_residues": len(mods), "unusual_occurrences": int(sum(mods.values()))}

    own = {e: v for e, v in edges.items() if v >= cfg.identity_threshold}
    group, _ = contract_hard_links(len(seqs), own, cfg.hard_link_identity)
    out["edges_at_config_threshold"] = len(own)
    out["nodes_after_hard_link_merge"] = int(group.max()) + 1

    bands = []
    for lo, hi in BANDS:
        pairs = [e for e, v in edges.items() if lo <= v < hi]
        cut = sum(bucket_of[i] != bucket_of[j] for i, j in pairs)
        bands.append({"band": f"{lo:.1f}-{min(hi, 1.0):.1f}" if hi <= 1 else f">={lo:.1f}", "pairs": len(pairs),
                      "cut": int(cut), "cut_%": round(100 * cut / len(pairs), 2) if pairs else None})
    allp = [e for e in edges]
    cut_all = sum(bucket_of[i] != bucket_of[j] for i, j in allp)
    bands.append({"band": f">={FLOOR:.1f} (all)", "pairs": len(allp), "cut": int(cut_all),
                  "cut_%": round(100 * cut_all / len(allp), 2) if allp else None})
    out["bands"] = bands

    aln = peps[peps["cluster_kind"] == "alignment"]
    sizes = aln.groupby("cluster_id").agg(peptides=("peptide_id", "size"), records=("n_rows", "sum")).sort_values("peptides", ascending=False)
    out["alignment_clusters"] = int(len(sizes))
    out["largest_alignment_cluster_peptides"] = int(sizes["peptides"].iloc[0])
    out["largest_alignment_cluster_%_records"] = round(100 * sizes["records"].iloc[0] / peps["n_rows"].sum(), 2)
    out["singleton_linear_peptides"] = int((peps["cluster_kind"] == "singleton").sum())
    out["murcko_clusters"] = int(peps.loc[peps["cluster_kind"] == "murcko", "cluster_id"].nunique())
    out["joint_clusters"] = int(peps["cluster_id"].nunique())

    smiles_buckets = peps.groupby("smiles")["bucket"].nunique()
    out["checks"] = {
        "clusters_spanning_buckets": int((peps.groupby("cluster_id")["bucket"].nunique() > 1).sum()),
        "identical_smiles_in_several_buckets": int((smiles_buckets > 1).sum()),
        "identical_token_sequences_in_several_buckets": 0,   # asserted above
        "pairs_at_or_above_hard_link_identity_across_buckets": int(sum(
            bucket_of[i] != bucket_of[j] for (i, j), v in edges.items() if v >= cfg.hard_link_identity)),
        "pairs_at_or_above_hard_link_identity": int(sum(v >= cfg.hard_link_identity for v in edges.values())),
    }
    b = peps.groupby("bucket").agg(peptides=("peptide_id", "size"), records=("n_rows", "sum"))
    b["nonlinear_%_of_records"] = (100 * peps[~peps["is_linear"]].groupby("bucket")["n_rows"].sum() / b["records"]).round(2)
    b["record_share_%"] = (100 * b["records"] / b["records"].sum()).round(2)
    if "n_active" in peps:
        b["active_%"] = (100 * peps.groupby("bucket")["n_active"].sum() / b["records"]).round(2)
    out["buckets"] = b.reset_index().to_dict("records")

    print(json.dumps(out, indent=1, default=str))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
