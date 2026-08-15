"""
Step 3: diff the full DBAASP raw pull against QMAP's included set.

Join key: DBAASP's own integer peptide `id`. This is the most reliable key
available -- QMAP's published dataset retains the original DBAASP id field
unchanged (verified: e.g. id=8 in both datasets has the identical sequence
"KVvvKWVvKvVK" / nTerminus C16 / cTerminus AMD), so no sequence-based fuzzy
matching is needed or preferable.

Buckets:
  a) in_both                         -- peptide id present in both pulls
  b1) excluded_non_monomer            -- QMAP filter: complexity != Monomer
  b2) excluded_unsupported_terminus   -- QMAP filter: N/C-term not in {None,ACT}/{None,AMD}
  b3) excluded_residue_no_smiles      -- QMAP filter: unresolved residue AND no native SMILES
  b4) excluded_nonstandard_bond       -- QMAP filter: bond type outside {DSB, AMD}
  b5) excluded_unexplained            -- passes our re-implementation of QMAP's filter but is
                                          still absent from QMAP's published set (data drift
                                          between QMAP's Jan-2026 snapshot and now, or a QMAP-side
                                          filter this re-implementation doesn't capture) -- flagged
                                          for manual review, not guessed
  c) in_qmap_not_in_raw_pull          -- present in QMAP but missing from our raw crawl

See qmap_filter.py's module docstring for the important finding that
"non-canonical residue WITH a native SMILES" is NOT, by itself, a QMAP exclusion
reason -- so there is no separate bucket for it here (it would fall through to
in_both or b5, not a residue-based bucket).
"""
import argparse
import os
import json
import csv
from collections import Counter

from dotenv import load_dotenv

from soamp.common.tracking import build_tracker
from soamp.curation.config import CurationConfig
from soamp.curation.parse_dbaasp import DBAASPPeptide
from soamp.curation.qmap_filter import qmap_inclusion_check
from soamp.utils.config import load_config


load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/curation/base.yaml")
    return parser.parse_args()


ARGS = _parse_args()
CFG = load_config(ARGS.config, CurationConfig)
RAW_JSONL = CFG.paths.cache_dir / "dbaasp_raw.jsonl"
QMAP_JSON = CFG.paths.cache_dir / "qmap_hf" / "dbaasp.json"
OUT_CSV = CFG.paths.data_dir / "dbaasp_vs_qmap_diff.csv"
LOG_PATH = CFG.paths.reports_dir / "step3_diff_log.txt"
TRACKER = build_tracker(CFG.tracking, CFG.paths.tracking_dir)


def main():
    TRACKER.log_config(CFG.model_dump(mode="json"))

    with open(QMAP_JSON) as f:
        qmap_data = json.load(f)
    qmap_ids = {e["id"]: e for e in qmap_data}

    raw_peptides = {}
    with open(RAW_JSONL) as f:
        for line in f:
            raw = json.loads(line)
            raw_peptides[raw["id"]] = DBAASPPeptide(raw)

    rows = []
    bucket_counter = Counter()
    bond_type_detail = Counter()
    terminus_detail = Counter()

    for pid, p in raw_peptides.items():
        if pid in qmap_ids:
            bucket = "in_both"
            reason = ""
        else:
            included, fail_reason = qmap_inclusion_check(p)
            if included:
                bucket = "b5_excluded_unexplained"
                reason = "passes reimplemented QMAP filter but absent from QMAP's published set"
            elif fail_reason == "non_monomer_complexity":
                bucket = "b1_excluded_non_monomer"
                reason = f"complexity={p.complexity}"
            elif fail_reason == "unsupported_nterminus":
                bucket = "b2_excluded_unsupported_terminus"
                reason = f"nterminus={p.nterminus}"
                terminus_detail[f"nterm:{p.nterminus}"] += 1
            elif fail_reason == "unsupported_cterminus":
                bucket = "b2_excluded_unsupported_terminus"
                reason = f"cterminus={p.cterminus}"
                terminus_detail[f"cterm:{p.cterminus}"] += 1
            elif fail_reason == "unresolved_residue_no_native_smiles":
                bucket = "b3_excluded_residue_no_smiles"
                codes = sorted({name for _, name in p.unusual_amino_acids})
                reason = f"unresolved non-canonical residue(s), no native SMILES; codes={codes}"
            elif fail_reason == "nonstandard_bond_type":
                bucket = "b4_excluded_nonstandard_bond"
                bts = sorted({(b.get("type") or {}).get("name") for b in p.intrachain_bonds
                              if (b.get("type") or {}).get("name") not in ("DSB", "AMD")})
                for bt in bts:
                    bond_type_detail[bt] += 1
                reason = f"bond type(s) outside DSB/AMD: {bts}"
            else:
                bucket = "b5_excluded_unexplained"
                reason = "unrecognized filter outcome"

        bucket_counter[bucket] += 1
        rows.append({
            "peptide_id": pid,
            "sequence": p.sequence,
            "has_native_smiles": p.has_native_smiles,
            "has_noncanonical": p.has_noncanonical,
            "bond_types": "|".join(p.bond_types),
            "nterminus": p.nterminus,
            "cterminus": p.cterminus,
            "complexity": p.complexity,
            "bucket": bucket,
            "reason": reason,
        })

    # bucket (c): in QMAP but not in our raw pull
    missing_from_raw = [pid for pid in qmap_ids if pid not in raw_peptides]
    for pid in missing_from_raw:
        e = qmap_ids[pid]
        rows.append({
            "peptide_id": pid,
            "sequence": e.get("sequence"),
            "has_native_smiles": bool(e.get("smiles")),
            "has_noncanonical": None,
            "bond_types": "|".join(sorted({b[2] for b in (e.get("bonds") or [])})),
            "nterminus": e.get("nterminal"),
            "cterminus": e.get("cterminal"),
            "complexity": None,
            "bucket": "c_in_qmap_not_in_raw_pull",
            "reason": "peptide id present in QMAP dataset but not returned by our DBAASP crawl "
                      "(fetch error, deleted/renumbered since QMAP's snapshot, or ID beyond our "
                      "crawl range)",
        })
    bucket_counter["c_in_qmap_not_in_raw_pull"] = len(missing_from_raw)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    TRACKER.log_artifact(
        name="dataset_curated", artifact_type="dataset",
        paths=[OUT_CSV],
        metadata={"n_raw_peptides": len(raw_peptides), "n_qmap_peptides": len(qmap_ids),
                  "bucket_counts": dict(bucket_counter)},
        depends_on=["raw_dbaasp", "raw_qmap"],
    )

    log_lines = []
    log_lines.append("=== Step 3: DBAASP raw pull vs QMAP included set -- diff ===")
    log_lines.append(f"Join key: DBAASP integer peptide id (native to both datasets)")
    log_lines.append(f"Raw pull peptides: {len(raw_peptides)}")
    log_lines.append(f"QMAP included peptides: {len(qmap_ids)}")
    log_lines.append("")
    log_lines.append("Bucket counts:")
    for b, cnt in sorted(bucket_counter.items()):
        log_lines.append(f"  {b}: {cnt}")
    log_lines.append("")
    log_lines.append("IMPORTANT METHODOLOGICAL FINDING (from reading QMAP's build_dataset.py "
                      "source directly): QMAP excludes a peptide for residue reasons ONLY when "
                      "an unresolved 'X' placeholder remains AND no native SMILES is available. "
                      "A non-canonical residue that already has a native DBAASP SMILES is NOT "
                      "excluded by QMAP for residue reasons -- so 'non-canonical WITH SMILES "
                      "available' is not a meaningful separate exclusion bucket; those peptides "
                      "are already in QMAP's baseline (bucket in_both) unless they also fail the "
                      "terminus or bond-type checks. The real recovery targets are bucket "
                      "b2 (terminus modifications like lipidation/PEGylation QMAP doesn't support) "
                      "and b4 (bond types like thioether/lactam/lactone cyclization QMAP excludes "
                      "unconditionally, regardless of SMILES availability).")
    log_lines.append("")
    log_lines.append(f"Bucket b2 terminus detail (which specific mod caused exclusion):")
    for t, cnt in terminus_detail.most_common():
        log_lines.append(f"  {t}: {cnt}")
    log_lines.append("")
    log_lines.append(f"Bucket b4 non-standard bond type detail:")
    for bt, cnt in bond_type_detail.most_common():
        log_lines.append(f"  {bt}: {cnt}")
    log_lines.append("")
    if missing_from_raw:
        log_lines.append(f"Bucket c examples (first 10, for manual inspection):")
        for pid in missing_from_raw[:10]:
            log_lines.append(f"  peptide_id={pid}, sequence={qmap_ids[pid].get('sequence')!r}")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print("\n".join(log_lines))
    TRACKER.close()


if __name__ == "__main__":
    main()
