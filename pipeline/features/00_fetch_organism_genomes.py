"""
Features step 0: fetch the RefSeq genome FASTA for every accession in
config/organism_genomes/organism_genome_accessions.csv.

This is what fills in raw_refseq_genomes -- prerequisite raw data for the
kmer_composition organism featurization method (see
src/soamp/features/organism_featurizers.py::KmerOrganismFeaturizer), so it
is numbered 00, ahead of the other features/ pipeline steps, the same way
curation's own crawl is numbered 01 ahead of everything that consumes it.

Reads the *reviewed* accessions CSV (run
pipeline/features/build_organism_genome_accessions_template.py first, and
review its output, before running this) and downloads each accession's
genomic FASTA into config/features/base.yaml's
organism_featurization.kmer_composition.genome_cache_dir
(default .cache/refseq_genomes/<accession>.fasta). Resumable: accessions
already cached on disk are skipped.

Output: .cache/refseq_genomes/<accession>.fasta (raw_refseq_genomes).
"""
import argparse
import csv

from dotenv import load_dotenv

from soamp.features.config import REPO_ROOT, FeaturesConfig
from soamp.features.genome_client import fetch_genome_fasta, make_session
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging

load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/features/base.yaml")
    return parser.parse_args()


ARGS = _parse_args()
CFG = load_config(ARGS.config, FeaturesConfig)
ACCESSIONS_CSV = REPO_ROOT / CFG.organism_featurization.kmer_composition.accessions_csv_path
GENOME_CACHE_DIR = REPO_ROOT / CFG.organism_featurization.kmer_composition.genome_cache_dir
LOG_PATH = CFG.paths.reports_dir / "features_step0_fetch_organism_genomes_log.txt"


def main() -> None:
    log = configure_logging("features.00_fetch_organism_genomes")

    with open(ACCESSIONS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    accessions = sorted({
        row["refseq_assembly_accession"] for row in rows if row["refseq_assembly_accession"]
    })

    GENOME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    fetched, skipped, failed = [], [], []
    for accession in accessions:
        out_path = GENOME_CACHE_DIR / f"{accession}.fasta"
        if out_path.exists():
            skipped.append(accession)
            continue
        try:
            fetch_genome_fasta(accession, out_path, session=session)
            fetched.append(accession)
        except Exception as e:
            failed.append((accession, str(e)))
            log.error(f"Failed to fetch {accession}: {e}")

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "=== Features step 0: fetch organism genomes ===",
        f"Accessions requested: {len(accessions)}",
        f"Fetched: {len(fetched)} {fetched}",
        f"Already cached (skipped): {len(skipped)} {skipped}",
        f"Failed: {len(failed)} {[a for a, _ in failed]}",
        f"Genome cache dir: {GENOME_CACHE_DIR}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))

    if failed:
        raise RuntimeError(f"failed to fetch {len(failed)} genome(s): {[a for a, _ in failed]}")


if __name__ == "__main__":
    main()
