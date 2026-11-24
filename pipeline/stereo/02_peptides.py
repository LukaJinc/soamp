"""
Stereo audit step 2: per-peptide table + notation-vs-SMILES disagreement file.

Reads residues.parquet / peptide_mapping.csv (step 1); writes peptides.csv and
disagreements.csv. Both sources keep their own label -- no winner is picked.
"""
import argparse
import json

import pandas as pd

from soamp.stereo.audit import peptide_label, residue_agreement
from soamp.stereo.config import StereoConfig
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/stereo/base.yaml")
    return p.parse_args()


def _counts(df: pd.DataFrame, stereo_col: str, prefix: str) -> pd.DataFrame:
    wide = (df.assign(k=prefix + "__" + df["residue_class"] + "__" + df[stereo_col].fillna("none"))
              .pivot_table(index="peptide_id", columns="k", values="position", aggfunc="count", fill_value=0))
    return wide


def _raw_meta(cfg) -> pd.DataFrame:
    rows = []
    with open(cfg.inputs.raw_jsonl) as f:
        for line in f:
            r = json.loads(line)
            rows.append({"peptide_id": r["id"], "n_intrachain_bonds": len(r.get("intrachainBonds") or []),
                         "n_interchain_bonds": len(r.get("interchainBonds") or []),
                         "nterminus": (r.get("nTerminus") or {}).get("name"),
                         "cterminus": (r.get("cTerminus") or {}).get("name")})
    return pd.DataFrame(rows)


def main() -> None:
    log = configure_logging("stereo.02_peptides")
    cfg = load_config(_parse_args().config, StereoConfig)
    out = cfg.outputs.dir
    res = pd.read_parquet(out / cfg.outputs.residues)
    mapping = pd.read_csv(out / "peptide_mapping.csv")
    processed = (pd.read_parquet(cfg.inputs.processed_parquet, columns=["peptide_id", "is_linear", "smiles", "sequence_full"])
                 .drop_duplicates("peptide_id"))

    res["agreement"] = res.apply(residue_agreement, axis=1)
    positional = res[res["smiles_stereo_corn"].notna()]
    corn_vs_tpl = positional["smiles_stereo_template"].isin(["L", "D", "achiral"]) & (
        positional["smiles_stereo_corn"] != positional["smiles_stereo_template"])
    res["corn_vs_template_differ"] = False
    res.loc[corn_vs_tpl[corn_vs_tpl].index, "corn_vs_template_differ"] = True
    res["notation_internal_conflict"] = res["notation_reason"].str.startswith("case says")

    grp = res.groupby("peptide_id")
    pep = mapping.merge(processed[["peptide_id", "is_linear"]], on="peptide_id", how="outer")
    pep["has_sequence"] = pep["length"].fillna(0) > 0
    pep["has_smiles"] = pep["peptide_id"].isin(processed.loc[processed["smiles"].notna(), "peptide_id"])
    pep = pep.merge(_raw_meta(cfg), on="peptide_id", how="left").set_index("peptide_id")
    pep = pep.join(_counts(res, "notation_stereo", "notation")).join(_counts(
        res[res["smiles_stereo_corn"].notna()], "smiles_stereo_corn", "smiles"))
    count_cols = [c for c in pep.columns if "__" in c]
    pep[count_cols] = pep[count_cols].fillna(0).astype(int)

    pep["label_notation"] = grp["notation_stereo"].agg(list).map(peptide_label)
    pep["label_notation_known_only"] = grp["notation_stereo"].agg(list).map(lambda s: peptide_label(s, True))
    pep["has_nonstandard"] = grp["residue_class"].agg(lambda s: bool((s == "nonstandard").any()))
    pep["has_D_notation"] = grp["notation_stereo"].agg(lambda s: bool((s == "D").any()))
    pep["has_D_nonstandard_notation"] = grp.apply(
        lambda g: bool(((g["residue_class"] == "nonstandard") & (g["notation_stereo"] == "D")).any()),
        include_groups=False)
    pos_ids = pep.index[pep["mapping_status"] == "positional"]
    smiles_stereo = grp["smiles_stereo_corn"].agg(list)
    pep["label_smiles"] = None
    pep.loc[pos_ids, "label_smiles"] = smiles_stereo.reindex(pos_ids).map(peptide_label)
    pep["has_D_smiles"] = None
    pep.loc[pos_ids, "has_D_smiles"] = smiles_stereo.reindex(pos_ids).map(lambda s: "D" in s)
    pep["is_cyclic"] = (pep["topology"] == "cyclic") | (pep["n_intrachain_bonds"].fillna(0) > 0)

    def agreement(row):
        pid = row.name
        if not row["has_sequence"]:
            return "no_sequence"
        if row["mapping_status"] == "positional":
            a = res_by_pid_agreement.get(pid, set())
            return "disagree" if "disagree" in a else ("agree" if a <= {"agree"} else "agree_partial_undetermined")
        if row["mapping_status"] in ("no_smiles", "smiles_parse_failed"):
            return "notation_only"
        sc = json.loads(row["smiles_counts"]) if isinstance(row["smiles_counts"], str) else None
        if sc and sum(sc.values()) == row["length"]:
            nc = {k: int(row.get(f"notation_total_{k}", 0)) for k in ("L", "D", "achiral")}
            return "counts_agree" if all(nc[k] == sc.get(k, 0) for k in nc) and not sc.get("undetermined") \
                else "counts_disagree"
        return "smiles_unmapped"

    res_by_pid_agreement = grp["agreement"].agg(set).to_dict()
    for k in ("L", "D", "achiral"):
        pep[f"notation_total_{k}"] = pep[[c for c in count_cols if c.startswith("notation__") and c.endswith(f"__{k}")]].sum(axis=1)
    pep["agreement_status"] = pep.apply(agreement, axis=1)
    pep = pep.drop(columns=["smiles_counts"]).reset_index()
    pep.to_csv(out / cfg.outputs.peptides, index=False)

    bad = res[(res["agreement"] == "disagree") | res["corn_vs_template_differ"] | res["notation_internal_conflict"]].copy()
    bad["why"] = (bad["agreement"].eq("disagree").map({True: "notation!=smiles_corn;"}).fillna("")
                  + bad["corn_vs_template_differ"].map({True: "smiles_corn!=smiles_template;"}).fillna("")
                  + bad["notation_internal_conflict"].map({True: "notation_case!=notation_name;"}).fillna(""))
    peptide_level = pep[pep["agreement_status"] == "counts_disagree"][["peptide_id"]].assign(
        why="peptide-level counts differ (positions could not be mapped)")
    pd.concat([bad[["peptide_id", "position", "char", "token", "residue_class", "case_stereo", "name_stereo",
                    "notation_stereo", "smiles_stereo_corn", "smiles_stereo_template", "why"]], peptide_level]
              ).to_csv(out / cfg.outputs.disagreements, index=False)
    log.info("peptides: %d | agreement: %s", len(pep), pep["agreement_status"].value_counts().to_dict())
    log.info("disagreement rows written: %d (+%d peptide-level)", len(bad), len(peptide_level))
    res.to_parquet(out / "residues_annotated.parquet", index=False)


if __name__ == "__main__":
    main()
