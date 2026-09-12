"""Client for NCBI's public Datasets v2 REST API
(https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/), used to resolve and
download a representative RefSeq genome assembly per bacterial species.

Plain `requests`, no NCBI SDK -- same style as
soamp.curation.dbaasp_client. Deliberately kept thin and not unit-tested
against the live API (same precedent as dbaasp_client.py): the pure k-mer
computation (soamp.features.kmer) and the featurizer logic carry the real
test coverage.
"""
import io
import zipfile

import requests

DATASET_REPORT_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/taxon/{taxon_id}/dataset_report"
DOWNLOAD_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{accession}/download"


def make_session() -> requests.Session:
    return requests.Session()


def resolve_reference_assembly(taxon_id: int, session: requests.Session | None = None,
                                timeout: int = 30) -> str | None:
    """Queries NCBI for genome assemblies under `taxon_id`, filtered to
    RefSeq reference/representative genomes, and returns the first
    accession found (None if no reference/representative assembly exists
    for this taxon). Reference genomes are unique per species by NCBI's own
    curation, so "first" is unambiguous when present.
    """
    session = session or requests.Session()
    params = {
        "filters.assembly_source": "refseq",
        "filters.reference_only": "true",
        "page_size": 1,
    }
    r = session.get(
        DATASET_REPORT_URL.format(taxon_id=taxon_id), params=params, timeout=timeout,
    )
    r.raise_for_status()
    reports = r.json().get("reports", [])
    if not reports:
        return None
    return reports[0]["accession"]


def fetch_genome_fasta(accession: str, out_path, session: requests.Session | None = None,
                        timeout: int = 120) -> None:
    """Downloads one assembly's genomic FASTA (the Datasets v2 download
    endpoint returns a zip bundle containing the sequence file plus
    reports) and writes the concatenated FASTA content to `out_path`.
    """
    session = session or requests.Session()
    params = {"include_annotation_type": "GENOME_FASTA"}
    r = session.get(
        DOWNLOAD_URL.format(accession=accession), params=params, timeout=timeout, stream=True,
    )
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        fasta_names = [n for n in zf.namelist() if n.endswith(".fna") or n.endswith(".fasta")]
        if not fasta_names:
            raise ValueError(f"no FASTA file found in download bundle for {accession!r}")
        with open(out_path, "wb") as out_f:
            for name in fasta_names:
                out_f.write(zf.read(name))
