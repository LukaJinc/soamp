"""
Step 2a: crawl the full DBAASP peptide ID space via the public REST API.

Writes one JSON object per line (native DBAASP peptide-card payload) to
.cache/dbaasp_raw.jsonl, plus a plain-text summary log to
.cache/dbaasp_crawl_log.txt.

This is the raw pull; flattening into data/dbaasp_raw_full.csv with the
audit columns happens in 02_parse_dbaasp_raw.py.
"""
import os

from soamp.curation.dbaasp_client import crawl_all

if __name__ == "__main__":
    out_jsonl = os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "dbaasp_raw.jsonl")
    log_path = os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "dbaasp_crawl_log.txt")
    crawl_all(out_jsonl, log_path, max_workers=20, resume=True)
