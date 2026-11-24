"""Does every organism of a dataset have a real (non-zero) organism representation?

Training loads the organism artifact without re-fitting it, and the genome-derived
featurizers (k-mer, DNABERT-S) encode an organism they cannot resolve as an all-zero
vector, silently. So before anything is rented, the launcher checks that each prebuilt
organism artifact the request needs (``soamp.runs.request.PREBUILT_ORGANISM``) exists
locally and resolves every organism in the dataset's classification CSV.

``vocab`` is built from the dataset itself on the instance if missing, so it always passes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from soamp.features.organism_featurizers import organism_featurizer_from_artifact, resolve_species_then_genus
from soamp.runs.request import DATASETS, ORGANISM_BUILD_HINT, ExperimentRequest
from soamp.utils.config import REPO_ROOT


def dataset_organisms(dataset: str, repo_root: Path = REPO_ROOT) -> list[str]:
    """Unique organism strings of the dataset's classification CSV (its first required input)."""
    path = repo_root / DATASETS[dataset]["required_artifacts"][0]
    with open(path, newline="") as f:
        return sorted({row["organism"].strip() for row in csv.DictReader(f)})


def unresolved_organisms(artifact_path: Path, organisms: list[str]) -> list[str]:
    """Organisms the artifact cannot represent (would be encoded as all zeros)."""
    featurizer = organism_featurizer_from_artifact(json.loads(artifact_path.read_text()))
    resolve = getattr(featurizer, "_resolve", None)
    if resolve is None:  # vocab: every organism it was built from has an index
        vocab = getattr(featurizer, "vocab", None) or {}
        return [o for o in organisms if o not in vocab]
    return [o for o in organisms if resolve(o) is None]


def source_unresolved(rep: str, organisms: list[str], repo_root: Path = REPO_ROOT) -> list[str]:
    """Organisms the representation's SOURCE table cannot resolve, i.e. what has to be added
    before the per-dataset artifact can be built: the genome accessions CSV for k-mer, the
    committed 16S embedding table for DNABERT-S. Empty when the source is unreadable."""
    species: dict[str, list[float]] = {}
    genus: dict[str, list[float]] = {}
    try:
        if rep == "kmer":
            with open(repo_root / "config/organism_genomes/organism_genome_accessions.csv", newline="") as f:
                for row in csv.DictReader(f):
                    if (row.get("refseq_assembly_accession") or "").strip():
                        (species if row["level"].strip() == "species" else genus)[row["match_key"].strip()] = []
        elif rep == "dnabert":
            table = json.loads((repo_root / "data/organism_16s_dnabert_s.json").read_text())
            for key, entry in table["entries"].items():
                (species if entry["level"] == "species" else genus)[key] = []
        else:
            return []
    except (OSError, KeyError, ValueError):
        return []
    return [o for o in organisms if resolve_species_then_genus(o, species, genus) is None]


def coverage_problems(req: ExperimentRequest, repo_root: Path = REPO_ROOT) -> list[str]:
    """One message per prebuilt organism representation that is missing or incomplete."""
    ds = DATASETS[req.dataset]
    prebuilt = ds.get("prebuilt", {}).get("organism", {})
    wanted = [o for o in req.representations.organism if o in prebuilt]
    if not wanted:
        return []
    try:
        organisms = dataset_organisms(req.dataset, repo_root)
    except FileNotFoundError as e:
        return [f"cannot check organism coverage: {e.filename} not found locally"]
    problems: list[str] = []
    for rep in wanted:
        hint = ORGANISM_BUILD_HINT[rep].format(dataset=req.dataset)
        for rel in prebuilt[rep]:
            path = repo_root / rel
            if not path.is_file():
                todo = source_unresolved(rep, organisms, repo_root)
                which = f" ({len(todo)} organisms not in its source table yet: {todo})" if todo else ""
                problems.append(f"organism representation {rep!r}: {rel} does not exist{which}. Build it on the laptop: {hint}")
                continue
            missing = unresolved_organisms(path, organisms)
            if missing:
                problems.append(
                    f"organism representation {rep!r}: {rel} has no vector for {len(missing)} of "
                    f"{len(organisms)} organisms of dataset {req.dataset!r} (they would train as all-zero "
                    f"inputs): {missing}. To fix: {hint}"
                )
    return problems
