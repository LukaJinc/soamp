"""Instance-side: download the W&B data artifact into the repo root.

The artifact stores files under their repo-relative paths (data/dev/...), as
written by scripts/exp/publish_data.py, so `download(root=...)` restores them.
usage: python scripts/exp/fetch_data.py <entity/project/name:alias>
"""
import sys

import wandb

from soamp.utils.config import REPO_ROOT


def main() -> int:
    ref = sys.argv[1]
    wandb.Api().artifact(ref).download(root=str(REPO_ROOT))
    print(f"[fetch_data] restored {ref} into {REPO_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
