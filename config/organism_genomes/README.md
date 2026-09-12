# Organism genome accessions

`organism_genome_accessions.csv` is the hand-reviewable `raw_refseq_genomes`
artifact: for each organism (or genus, as a fallback), the specific RefSeq
genome assembly accession used to compute its k-mer composition organism
representation (`kmer_composition` in `organism_featurization.method`, see
`src/soamp/features/organism_featurizers.py::KmerOrganismFeaturizer`).

## Columns

- `match_key` -- the organism string (species row, e.g. `"Escherichia
  coli"`, matched exactly against the dataset's `organism` column) or the
  genus alone (genus row), same precedence convention as
  `config/thresholds/organism_thresholds.csv`.
- `level` -- `species` or `genus`.
- `ncbi_taxon_id` -- the NCBI Taxonomy ID used to resolve a reference
  assembly. For species already in `config/thresholds/organism_thresholds.csv`,
  reuse the taxon ID already recorded there (`ncbi_taxon_id_if_available`)
  rather than re-deriving it.
- `refseq_assembly_accession` -- the specific RefSeq assembly accession
  (e.g. `GCF_000005845.2`) whose genomic FASTA is downloaded and k-mer'd.
  **Pinned at curation time, not re-resolved automatically on every
  pipeline run** — NCBI's "reference genome" designation for a species can
  change over time; pinning keeps results reproducible. Auto-resolved
  candidates come from `pipeline/features/build_organism_genome_accessions_template.py`
  (queries NCBI's Datasets v2 API for the current reference/representative
  assembly) but are meant to be reviewed, not blindly trusted.
- `source` -- how the accession was obtained (e.g. `"NCBI Datasets v2 API,
  reference_only=true, YYYY-MM-DD"`).
- `notes` -- free text.

## Lookup precedence

Same as `organism_thresholds.csv`: exact `species`-level match wins,
otherwise a `genus`-level match on the organism's first word, otherwise
unmatched (the featurizer falls back to an all-zero vector for an
unmatched organism at inference — see `KmerOrganismFeaturizer.encode`).

## Updating

Run `pipeline/features/build_organism_genome_accessions_template.py` to
auto-resolve candidate accessions for any organism in
`organism_thresholds.csv` that (a) has a filled `ncbi_taxon_id_if_available`
**and** (b) actually has thresholds filled in (`active_threshold_uM`/
`inactive_threshold_uM`) — i.e. is in `dataset_model_ready`'s current v1
scope, not merely present in the 499-row threshold template. This keeps
NCBI traffic and the genome cache scoped to organisms the model actually
trains/evaluates on. It never overwrites an existing row, only adds new
ones. Review new rows before running
`pipeline/features/00_fetch_organism_genomes.py`, which downloads whatever
accessions are recorded here.
