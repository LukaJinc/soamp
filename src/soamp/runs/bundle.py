"""Code bundle: the tarball an instance downloads from W&B instead of being rsynced.

Only code and config ship (the tracked tree is ~265 MB, nearly all of it data/ and
reports/ CSVs). Data travels separately in the ``soamp_<dataset>_data`` artifact.
Generated per-cell configs/specs are untracked, so callers pass them via ``extra``.
"""
from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

# Everything an instance needs to run pipeline/train_cv.py + scripts/vast/run_experiments.py.
INCLUDE_PREFIXES = ("src/", "pipeline/", "config/", "scripts/vast/", "scripts/exp/")
INCLUDE_FILES = ("pyproject.toml", "README.md", "scripts/colab/requirements-colab.txt")


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\0") if f]


def select_files(files: list[str], extra: list[str] = ()) -> list[str]:
    keep = [f for f in files if f.startswith(INCLUDE_PREFIXES) or f in INCLUDE_FILES]
    return sorted(set(keep) | set(extra))


def build_tarball(root: Path, files: list[str], out: Path) -> Path:
    missing = [f for f in files if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(f"bundle files missing on disk: {missing[:5]}")
    with tarfile.open(out, "w:gz") as tar:
        for f in files:
            tar.add(root / f, arcname=f)
    return out
