"""Provenance for the W&B data artifact (pure logic; scripts/exp/publish_data.py does the upload).

An instance restores ``soamp_<dataset>_data`` and trusts it, so the artifact says what it holds: per-file
size and sha256, the dataset's size and label counts, the strains, the split, and the git commit it was
built from. A later download can be checked file by file against this metadata.
"""
from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_state(root: Path) -> dict:
    """{'sha': HEAD commit or None, 'dirty': True if tracked files differ from it}."""
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def dataset_counts(classification_csv: Path) -> dict:
    """Row/peptide/label/strain/split counts of a ``mic_classification_dataset.csv``."""
    labels: dict[str, int] = {}
    splits: dict[str, int] = {}
    organisms: dict[str, int] = {}
    peptides: set[str] = set()
    rows = 0
    with open(classification_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows += 1
            peptides.add(r["peptide_id"])
            labels[r["label"]] = labels.get(r["label"], 0) + 1
            splits[r["split"]] = splits.get(r["split"], 0) + 1
            organisms[r["organism"]] = organisms.get(r["organism"], 0) + 1
    return {"labelled_pairs": rows, "peptides": len(peptides), "labels": dict(sorted(labels.items())),
            "split_rows": dict(sorted(splits.items())), "pairs_per_organism": dict(sorted(organisms.items()))}


def build_metadata(dataset: str, root: Path, present_files: list[str], required_csv: str) -> dict:
    """Metadata attached to the artifact. ``required_csv`` is the repo-relative classification CSV."""
    files = {f: {"bytes": (root / f).stat().st_size, "sha256": sha256_of(root / f)} for f in present_files}
    return {
        "dataset": dataset,
        "git": git_state(root),
        "counts": dataset_counts(root / required_csv),
        "files": files,
        "note": "Instances restore every file under its repo-relative path with scripts/exp/fetch_data.py.",
    }


def describe(meta: dict) -> str:
    c = meta["counts"]
    return (f"{meta['dataset']}: {c['labelled_pairs']} labelled pairs, {c['peptides']} peptides, labels {c['labels']}, "
            f"{len(c['pairs_per_organism'])} strains, {len(meta['files'])} files; git {str(meta['git']['sha'])[:8]}"
            f"{' (uncommitted changes)' if meta['git']['dirty'] else ''}")


def verify_download(meta: dict, root: Path) -> list[str]:
    """Problems found when comparing restored files under ``root`` with the artifact metadata (empty = all good)."""
    problems = []
    for f, info in meta["files"].items():
        p = root / f
        if not p.is_file():
            problems.append(f"missing: {f}")
        elif p.stat().st_size != info["bytes"]:
            problems.append(f"size differs: {f}")
        elif sha256_of(p) != info["sha256"]:
            problems.append(f"sha256 differs: {f}")
    return problems
