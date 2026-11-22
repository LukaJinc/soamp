"""List vast.ai offers matching a run spec, MOST RELIABLE FIRST (read-only).

Reliability, not price, ranks the list: a host dropping mid-run loses the
in-flight cell. Price is shown only as a tie-break. Re-run at launch time;
offers change, so never hardcode an offer id.

usage: python scripts/vast/find_offers.py --spec config/vast/dev_peptideclm_vocab.yaml [--top 8]
"""
import argparse
import json
import subprocess
import sys

from soamp.runs.spec import RunSpec, offer_query
from soamp.utils.config import REPO_ROOT, load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    query = offer_query(load_config(args.spec, RunSpec).instance)
    out = subprocess.run(
        [str(REPO_ROOT / ".venv/bin/vastai"), "search", "offers", query, "-o", "reliability-", "--raw"],
        capture_output=True, text=True, check=True,
    ).stdout
    offers = json.loads(out)[: args.top]
    sys.stdout.write(f"{'id':>9} {'gpu':<12} {'$/hr':>6} {'reliab':>7} {'down':>6}  location\n")
    for o in offers:
        sys.stdout.write(
            f"{o['id']:>9} {o['gpu_name']:<12} {o['dph_total']:>6.2f} {o['reliability2']:>7.4f} "
            f"{o['inet_down']:>6.0f}  {o['geolocation']}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
