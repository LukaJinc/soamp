"""Generate/refresh config/organism_genomes/organism_genome_accessions.csv.

Not a numbered pipeline step -- a maintenance utility for the hand-reviewable
`raw_refseq_genomes` artifact, mirroring
pipeline/labeling/build_threshold_template.py's own "auto-seed, never
overwrite what's been reviewed" discipline.

For every species row in config/thresholds/organism_thresholds.csv that has
a filled `ncbi_taxon_id_if_available` and isn't already present in
organism_genome_accessions.csv, queries NCBI's Datasets v2 API
(soamp.features.genome_client.resolve_reference_assembly) for a candidate
RefSeq reference assembly accession and appends a new row. Rows already in
the CSV are left untouched -- the accession is pinned at review time, not
re-resolved on every run (see config/organism_genomes/README.md).
"""
import csv
from datetime import date

from soamp.features.config import REPO_ROOT, FeaturesConfig
from soamp.features.genome_client import resolve_reference_assembly
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging

CFG = load_config("config/features/base.yaml", FeaturesConfig)
THRESHOLDS_CSV = REPO_ROOT / "config" / "thresholds" / "organism_thresholds.csv"
ACCESSIONS_CSV = REPO_ROOT / CFG.organism_featurization.kmer_composition.accessions_csv_path

FIELDNAMES = ["match_key", "level", "ncbi_taxon_id", "refseq_assembly_accession", "source", "notes"]


def main() -> None:
    log = configure_logging("build_organism_genome_accessions_template")

    with open(THRESHOLDS_CSV, newline="") as f:
        threshold_rows = list(csv.DictReader(f))

    # Scoped to organisms that actually have thresholds filled in (matching
    # dataset_model_ready's current v1 scope -- E. coli/S. aureus/
    # P. aeruginosa), not every species row with a taxon ID (499 of them) --
    # a genome fetch/k-mer computation for an organism the model never
    # trains or evaluates on would be wasted NCBI traffic and cache space.
    candidates = [
        (row["match_key"], row["level"], row["ncbi_taxon_id_if_available"])
        for row in threshold_rows
        if row["level"] == "species"
        and row["ncbi_taxon_id_if_available"]
        and (row["active_threshold_uM"] or row["inactive_threshold_uM"])
    ]

    existing: dict[tuple[str, str], dict] = {}
    if ACCESSIONS_CSV.exists():
        with open(ACCESSIONS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                existing[(row["level"], row["match_key"])] = row

    out_rows: list[dict] = list(existing.values())
    new_keys, unresolved_keys = [], []
    today = date.today().isoformat()

    for match_key, level, taxon_id in candidates:
        key = (level, match_key)
        if key in existing:
            continue
        accession = resolve_reference_assembly(int(taxon_id))
        if accession is None:
            unresolved_keys.append(match_key)
            continue
        out_rows.append({
            "match_key": match_key,
            "level": level,
            "ncbi_taxon_id": taxon_id,
            "refseq_assembly_accession": accession,
            "source": f"NCBI Datasets v2 API, reference_only=true, {today}",
            "notes": "",
        })
        new_keys.append(match_key)

    ACCESSIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(ACCESSIONS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    log.info(
        f"Wrote {ACCESSIONS_CSV}: {len(out_rows)} total rows, "
        f"{len(new_keys)} newly resolved ({new_keys}), "
        f"{len(unresolved_keys)} unresolved ({unresolved_keys})"
    )


if __name__ == "__main__":
    main()
