"""
Stereo audit step 1: per-residue classification (notation vs SMILES).

Read-only on all existing data. For every dev-corpus peptide, classifies each
sequence position by residue class (standard / nonstandard / unrecognized) and
stereo class (L / D / achiral / undetermined) two independent ways -- from the
DBAASP notation and from the SMILES (backbone-role chirality + template match,
no CIP). Output: <outputs.dir>/residues.parquet. Nothing is dropped; what cannot
be classified is 'undetermined' with a reason.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from soamp.stereo.audit import audit_peptide
from soamp.stereo.config import StereoConfig
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/stereo/base.yaml")
    return p.parse_args()


def _work(chunk):
    cache, residues, infos = {}, [], []
    for pid, seq, smi, unusual in chunk:
        rows, info = audit_peptide(pid, seq, smi, unusual, cache)
        residues += rows
        infos.append(info)
    return residues, infos


def main() -> None:
    log = configure_logging("stereo.01_residues")
    cfg = load_config(_parse_args().config, StereoConfig)
    smiles = (pd.read_parquet(cfg.inputs.processed_parquet, columns=["peptide_id", "smiles"])
              .drop_duplicates("peptide_id").set_index("peptide_id")["smiles"])
    items = []
    with open(cfg.inputs.raw_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in smiles.index:
                items.append((r["id"], r.get("sequence"), smiles[r["id"]], r.get("unusualAminoAcids")))
    log.info("%d dev peptides found in the raw file (of %d with SMILES)", len(items), len(smiles))
    chunks = [items[i::cfg.n_jobs * 8] for i in range(cfg.n_jobs * 8)]
    residues, infos = [], []
    with ProcessPoolExecutor(cfg.n_jobs) as pool:
        for res, inf in pool.map(_work, chunks):
            residues += res
            infos += inf
    out = cfg.outputs.dir
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(residues)
    df.to_parquet(out / cfg.outputs.residues, index=False)
    info_df = pd.DataFrame(infos)
    info_df["smiles_counts"] = info_df["smiles_counts"].map(lambda c: json.dumps(dict(c)) if c else None)
    info_df.to_csv(out / "peptide_mapping.csv", index=False)
    missing = set(smiles.index) - {i[0] for i in items}
    log.info("residues: %d rows; mapping status: %s; dev peptides absent from raw: %d",
             len(df), info_df["mapping_status"].str.split("(").str[0].value_counts().to_dict(), len(missing))


if __name__ == "__main__":
    main()
