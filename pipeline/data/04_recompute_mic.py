"""
Dev v2 step 1: recompute every MIC in ug/ml from ``concentration`` (+ unit + RDKit MW).

Reads data/dev/dbaasp_dev_processed.parquet (read-only), keeps the target strains,
applies soamp.data.mic_recompute, drops the rows that cannot be recomputed (after
checking their number against the config), and writes

  data/dev_v2/dbaasp_dev_recomputed.parquet     every kept row, with mic_ug_ml, direction, conc_kind, ...
  data/dev_v2/recompute_dropped_rows.csv        the dropped rows and why
  reports/dev_v2/dbaasp_bogus_mw_peptides.csv   peptides whose DBAASP activity implies an unusable MW
  reports/dev_v2/mic_recompute_report.md        counts, agreement with DBAASP's activity, NA rows
"""
import argparse

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from soamp.data.mic_recompute import NA_SENTINEL, recompute_mic
from soamp.data.recompute_config import RecomputeConfig
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging

load_dotenv()


def _md(df: pd.DataFrame) -> str:
    head = [str(df.index.name or "")] + [str(c) for c in df.columns]
    rows = [[str(i)] + [str(v) for v in r] for i, r in zip(df.index, df.itertuples(index=False))]
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


def build_report(out: pd.DataFrame, dropped: pd.DataFrame, bogus: pd.DataFrame, cfg: RecomputeConfig) -> str:
    ok = out[out["recompute_status"] == "ok"].copy()
    ok["unit_n"] = ok["unit"].map(lambda u: "uM" if "M" in str(u).upper() and "G" not in str(u).upper() else "ug/ml")
    ok["within 0.1%"] = (ok["activity_ratio"] - 1).abs() <= 1e-3
    ok["within 1%"] = (ok["activity_ratio"] - 1).abs() <= 1e-2
    agree = ok.groupby(["conc_kind", "unit_n"]).agg(rows=("activity_ratio", "size"),
                                                    within_0p1pct=("within 0.1%", "mean"),
                                                    within_1pct=("within 1%", "mean"),
                                                    median_ratio=("activity_ratio", "median"))
    agree[["within_0p1pct", "within_1pct"]] = (100 * agree[["within_0p1pct", "within_1pct"]]).round(2)
    agree["median_ratio"] = agree["median_ratio"].round(4)
    na = out[out["conc_kind"] == "NA"]
    lines = [
        "# MIC recomputation report (dev v2, seven strains)", "",
        f"Input rows in scope: {len(out) + len(dropped)}; kept: {len(out)}; dropped: {len(dropped)}.", "",
        "## Rule", "ug/ml rows: the concentration as written. uM rows: uM x RDKit MW / 1000. "
        f"NA rows: the DBAASP sentinel {NA_SENTINEL:,.0f}. Anything else is dropped, not guessed.", "",
        "## Concentration kinds (all kept rows)", _md(out["conc_kind"].value_counts().to_frame("rows")), "",
        "## Recomputed vs DBAASP `activity` (rows with a value; NA rows excluded)",
        "Ratio = activity / recomputed. uM rows differ by a small MW offset (DBAASP's weight vs RDKit's); "
        "ug/ml rows should match.", _md(agree), "",
        f"## NA rows\n{len(na)} rows, all set to {NA_SENTINEL:,.0f}; measures: {na['measure'].value_counts(dropna=False).to_dict()}; "
        f"with a unit: {int(na['unit'].notna().sum())}.", "",
        f"## Dropped rows ({len(dropped)})", _md(dropped[["peptide_id", "measure", "concentration", "unit", "recompute_status"]]
                                                 .reset_index(drop=True)) if len(dropped) else "_none_", "",
        f"## Bogus molecular weights\nPeptides with at least one uM row whose DBAASP activity implies a weight below "
        f"{cfg.bogus_mw_below:g} g/mol: {len(bogus)} (list in `{cfg.bogus_mw_filename}`).",
        _md(bogus.head(15).reset_index(drop=True)) if len(bogus) else "_none_", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data/recompute_dev_v2.yaml")
    cfg = load_config(parser.parse_args().config, RecomputeConfig)
    log = configure_logging("data.04_recompute_mic")

    raw = pd.read_parquet(cfg.resolve(cfg.paths.processed_parquet))
    scope = raw[raw["target_species"].isin(cfg.organisms)]
    missing = sorted(set(cfg.organisms) - set(scope["target_species"]))
    if missing:
        raise SystemExit(f"organisms with no rows: {missing}")
    log.info("%d rows for %d strains", len(scope), len(cfg.organisms))

    table = recompute_mic(scope)
    bad = table["recompute_status"].str.startswith("dropped")
    dropped, kept = table[bad], table[~bad].copy()
    if len(dropped) != cfg.expected_dropped_rows:
        raise SystemExit(f"{len(dropped)} rows cannot be recomputed, config expects {cfg.expected_dropped_rows}: "
                         f"{dropped['recompute_status'].value_counts().to_dict()}")
    log.info("dropped %d rows: %s", len(dropped), dropped["recompute_status"].value_counts().to_dict())

    low = kept[kept["implied_mw"] < cfg.bogus_mw_below]
    bogus = (low.groupby(["peptide_id", "dbaaspId", "name"], dropna=False)
                .agg(rows=("implied_mw", "size"), implied_mw=("implied_mw", lambda s: ",".join(f"{v:g}" for v in sorted(set(s.round(2))))),
                     rdkit_mw=("mw", "first")).reset_index())
    bogus["rdkit_mw"] = bogus["rdkit_mw"].round(2)

    out_dir, rep_dir = cfg.resolve(cfg.paths.out_dir), cfg.resolve(cfg.paths.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out_dir / cfg.recomputed_filename, index=False)
    dropped.to_csv(out_dir / cfg.dropped_filename, index=False)
    bogus.to_csv(rep_dir / cfg.bogus_mw_filename, index=False)
    (rep_dir / cfg.report_filename).write_text(build_report(kept, dropped, bogus, cfg))
    log.info("wrote %s (%d rows), %d bogus-MW peptides", out_dir / cfg.recomputed_filename, len(kept), len(bogus))


if __name__ == "__main__":
    main()
