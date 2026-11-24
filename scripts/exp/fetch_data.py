"""Download the W&B data artifact into the repo root (instances) or any folder (checks).

The artifact stores files under their repo-relative paths (data/dev_v2/...), as written by
scripts/exp/publish_data.py, so `download(root=...)` restores them. When the artifact carries file
checksums (every artifact published since the provenance metadata was added), each restored file is
verified against them and a mismatch fails the run (exit 1) instead of training on bad data.

usage: python scripts/exp/fetch_data.py <entity/project/name:alias> [target_dir]
  target_dir defaults to the repo root; pass a temp dir to verify an upload without touching data/.
"""
import sys
from pathlib import Path

import wandb
from dotenv import load_dotenv

from soamp.runs.data_artifact import verify_download
from soamp.utils.config import REPO_ROOT


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")   # laptop: the API key lives in .env; on an instance there is no .env (no-op)
    ref = sys.argv[1]
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO_ROOT
    art = wandb.Api().artifact(ref)
    art.download(root=str(root))
    print(f"[fetch_data] restored {ref} into {root}")
    if "files" in (art.metadata or {}):
        problems = verify_download(art.metadata, root)
        if problems:
            print(f"[fetch_data] CHECKSUM PROBLEMS: {problems}")
            return 1
        print(f"[fetch_data] verified {len(art.metadata['files'])} files against the artifact checksums")
    else:
        print("[fetch_data] artifact has no checksum metadata (older publish): not verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
