"""
Features step 4: embed each organism's 16S rRNA gene with frozen DNABERT-S.

For every row of config/organism_genomes/organism_genome_accessions.csv:
  1. make sure the assembly's genome FASTA (step 0's cache) and GFF3
     annotation (fetched here, cached) are on disk,
  2. cut out the full-length 16S rRNA gene (soamp.features.dna_sequence),
  3. embed it with DNABERT-S (soamp.features.dnabert_s, mean-pooled, 768-d).
The result is written as a small table (config
organism_16s_embedding.output_path, default data/organism_16s_dnabert_s.json)
that the `dnabert_s_16s` organism featurizer reads at training time.

Run this ONCE, on Colab (scripts/colab/04_embed_organisms.ipynb): the
checkpoint's remote code needs `transformers<5`, `einops` and `triton`, none
of which the main environment has. Commit the table; re-run only when the
accessions CSV gains organisms.
"""
import argparse
import csv
import hashlib
import json

from dotenv import load_dotenv

from soamp.features.config import REPO_ROOT, FeaturesConfig
from soamp.features.dna_sequence import extract_16s, parse_fasta_records, parse_gff_rrna_features
from soamp.features.dnabert_s import embed_sequences
from soamp.features.genome_client import fetch_genome_fasta, fetch_genome_gff, make_session
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging

load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/features/base.yaml")
    return parser.parse_args()


ARGS = _parse_args()
CFG = load_config(ARGS.config, FeaturesConfig)
EMB = CFG.organism_16s_embedding
KMER = CFG.organism_featurization.kmer_composition
ACCESSIONS_CSV = REPO_ROOT / KMER.accessions_csv_path
GENOME_CACHE_DIR = REPO_ROOT / KMER.genome_cache_dir
GFF_CACHE_DIR = REPO_ROOT / EMB.gff_cache_dir
OUT_JSON = REPO_ROOT / EMB.output_path
LOG_PATH = CFG.paths.reports_dir / "features_step4_embed_organism_16s_dnabert_s_log.txt"


def main() -> None:
    log = configure_logging("features.04_embed_organism_16s_dnabert_s")

    with open(ACCESSIONS_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["refseq_assembly_accession"]]
    accessions = sorted({r["refseq_assembly_accession"] for r in rows})
    log.info(f"{len(rows)} organism rows, {len(accessions)} assemblies")

    GENOME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GFF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    sequence_by_accession: dict[str, str] = {}
    for accession in accessions:
        fasta_path = GENOME_CACHE_DIR / f"{accession}.fasta"
        gff_path = GFF_CACHE_DIR / f"{accession}.gff"
        if not fasta_path.exists():
            fetch_genome_fasta(accession, fasta_path, session=session)
        if not gff_path.exists():
            fetch_genome_gff(accession, gff_path, session=session)
        sequence_by_accession[accession] = extract_16s(
            parse_fasta_records(fasta_path), parse_gff_rrna_features(gff_path)
        )
        log.info(f"{accession}: 16S rRNA {len(sequence_by_accession[accession])} bp")

    vectors = embed_sequences(
        [sequence_by_accession[a] for a in accessions],
        checkpoint=EMB.checkpoint, revision=EMB.revision, max_length=EMB.max_length,
        batch_size=EMB.batch_size, device=EMB.device,
    )
    vector_by_accession = {a: vectors[i].tolist() for i, a in enumerate(accessions)}

    table = {
        "method": "dnabert_s_16s",
        "checkpoint": EMB.checkpoint,
        "revision": EMB.revision,
        "embedding_dim": int(vectors.shape[1]),
        "sequence_source": "full-length 16S rRNA gene (GFF3 feature, longest copy) of the RefSeq assembly",
        "entries": {
            r["match_key"]: {
                "level": r["level"],
                "accession": r["refseq_assembly_accession"],
                "length_bp": len(sequence_by_accession[r["refseq_assembly_accession"]]),
                "sequence_sha256": hashlib.sha256(
                    sequence_by_accession[r["refseq_assembly_accession"]].encode()
                ).hexdigest(),
                "vector": vector_by_accession[r["refseq_assembly_accession"]],
            }
            for r in rows
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)
        f.write("\n")

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "=== Features step 4: DNABERT-S 16S organism embeddings ===",
        f"checkpoint: {EMB.checkpoint} @ {EMB.revision}",
        f"organisms: {sorted(table['entries'])}",
        f"embedding_dim: {table['embedding_dim']}",
        f"Wrote {OUT_JSON}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))


if __name__ == "__main__":
    main()
