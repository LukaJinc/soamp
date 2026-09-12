"""
Step 2a: crawl the full DBAASP peptide ID space via the public REST API.

Writes one JSON object per line (native DBAASP peptide-card payload) to
.cache/dbaasp_raw.jsonl, plus a plain-text summary log to
.cache/dbaasp_crawl_log.txt.

This is the raw pull; flattening into data/dbaasp_raw_full.csv with the
audit columns happens in 03_parse_dbaasp_raw.py.

Output: .cache/dbaasp_raw.jsonl (raw_dbaasp).
"""
import argparse

from dotenv import load_dotenv

from soamp.curation.config import CurationConfig
from soamp.curation.dbaasp_client import crawl_all
from soamp.utils.config import load_config


load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/curation/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    CFG = load_config(args.config, CurationConfig)
    out_jsonl = CFG.paths.cache_dir / "dbaasp_raw.jsonl"
    log_path = CFG.paths.cache_dir / "dbaasp_crawl_log.txt"

    n_ok, n_not_found, n_error = crawl_all(
        out_jsonl, log_path,
        max_workers=CFG.crawl.max_workers,
        max_id=CFG.crawl.max_id,
        probe_margin=CFG.crawl.id_margin,
        resume=True,
    )
    print(f"crawl done: n_ok={n_ok} n_not_found={n_not_found} n_error={n_error}")


if __name__ == "__main__":
    main()
