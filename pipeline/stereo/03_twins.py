"""
Stereo audit step 3: stereo-only twins.

Every D-residue is mapped to its L-form (canonical: upper-case letter; unusual:
name without the 'D-' prefix). Peptides that are identical after that mapping
but differ in notation stereo are twins. Reports the pairs, whether they also
differ in termini/bonds, MIC differences for shared organisms, and whether the
twins fall in different buckets of the current dev split.
"""
import argparse
import json

import numpy as np
import pandas as pd

from soamp.stereo.config import StereoConfig
from soamp.stereo.twins import to_l_token, twin_pairs
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging

MIC_SENTINEL = 1e6  # DBAASP 'NA' activity placeholder (1111111)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/stereo/base.yaml")
    return p.parse_args()


def main() -> None:
    log = configure_logging("stereo.03_twins")
    cfg = load_config(_parse_args().config, StereoConfig)
    out = cfg.outputs.dir
    res = pd.read_parquet(out / "residues_annotated.parquet").sort_values(["peptide_id", "position"])
    raw = {}
    with open(cfg.inputs.raw_jsonl) as f:
        for line in f:
            r = json.loads(line)
            bonds = tuple(sorted((b["position1"], b["position2"], (b.get("type") or {}).get("name"))
                                 for b in r.get("intrachainBonds") or []))
            raw[r["id"]] = ((r.get("nTerminus") or {}).get("name"), (r.get("cTerminus") or {}).get("name"), bonds)
    recs, skipped = [], 0
    for pid, g in res.groupby("peptide_id"):
        if (g["residue_class"] == "unrecognized").any():
            skipped += 1   # cannot be mapped without guessing; counted, not dropped silently
            continue
        recs.append({"peptide_id": pid,
                     "mapped": tuple(to_l_token(t, c) for t, c in zip(g["token"], g["char"])),
                     "stereo": tuple(g["notation_stereo"]), "annotation": raw.get(pid)})
    pairs = pd.DataFrame(twin_pairs(recs))
    log.info("%d peptides mapped, %d skipped (unrecognized residue); %d twin pairs", len(recs), skipped, len(pairs))
    pairs.to_csv(out / cfg.outputs.twins, index=False)
    if pairs.empty:
        return

    proc = pd.read_parquet(cfg.inputs.processed_parquet, columns=["peptide_id", "target_species", "measure", "activity", "unit"])
    mic = proc[(proc["measure"] == "MIC") & proc["activity"].between(1e-9, MIC_SENTINEL, inclusive="left")]
    per = mic.groupby(["peptide_id", "target_species"])["activity"].median().rename("mic").reset_index()
    merged = (pairs.merge(per.add_suffix("_a"), left_on="peptide_a", right_on="peptide_id_a")
                   .merge(per.add_suffix("_b"), left_on=["peptide_b", "target_species_a"],
                          right_on=["peptide_id_b", "target_species_b"]))
    merged["abs_log2_mic_ratio"] = np.abs(np.log2(merged["mic_a"] / merged["mic_b"]))
    merged = merged.rename(columns={"target_species_a": "organism"})[
        ["peptide_a", "peptide_b", "same_annotation", "organism", "mic_a", "mic_b", "abs_log2_mic_ratio"]]
    merged.to_csv(out / cfg.outputs.twin_mic, index=False)

    split = pd.read_parquet(cfg.inputs.split_parquet).set_index("peptide_id")
    bucket = split["split"].astype(str) + "/" + split["fold_id"].astype(str)
    a, b = pairs["peptide_a"].map(bucket), pairs["peptide_b"].map(bucket)
    pairs["bucket_a"], pairs["bucket_b"] = a, b
    pairs["different_bucket"] = np.where(a.isna() | b.isna(), None, a != b)
    pairs.to_csv(out / cfg.outputs.twins, index=False)
    log.info("pairs with a shared-organism MIC: %d | in different buckets: %s | not in split: %d",
             merged[["peptide_a", "peptide_b"]].drop_duplicates().shape[0],
             int((pairs["different_bucket"] == True).sum()), int(a.isna().sum() + b.isna().sum()))  # noqa: E712


if __name__ == "__main__":
    main()
