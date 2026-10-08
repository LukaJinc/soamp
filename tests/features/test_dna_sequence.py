import pytest

from soamp.features.dna_sequence import (
    DnaSequenceError,
    RrnaFeature,
    extract_16s,
    parse_fasta_records,
    parse_gff_rrna_features,
    reverse_complement,
)


def _gene(length, base="ACGT"):
    return (base * (length // len(base) + 1))[:length]


def test_parse_fasta_records_multi_record_multi_line(tmp_path):
    path = tmp_path / "g.fasta"
    path.write_text(">chr1 desc\nacgt\nTTGA\n\n>plasmid\nGGCC\n")
    assert parse_fasta_records(path) == [("chr1 desc", "ACGTTTGA"), ("plasmid", "GGCC")]


def test_parse_gff_rrna_features_reads_product_and_skips_other_types(tmp_path):
    path = tmp_path / "g.gff"
    path.write_text(
        "##gff-version 3\n"
        "chr1\tRefSeq\tgene\t1\t100\t.\t+\t.\tID=gene1\n"
        "chr1\tRefSeq\trRNA\t10\t1551\t.\t-\t.\tID=r1;product=16S ribosomal RNA\n"
        "chr1\tRefSeq\trRNA\t2000\t2119\t.\t+\t.\tID=r2;product=5S ribosomal RNA\n"
    )
    features = parse_gff_rrna_features(path)
    assert features == [
        RrnaFeature("chr1", 10, 1551, "-", "16S ribosomal RNA"),
        RrnaFeature("chr1", 2000, 2119, "+", "5S ribosomal RNA"),
    ]


def test_reverse_complement():
    assert reverse_complement("AACGT") == "ACGTT"


def test_extract_16s_plus_strand_uses_one_based_inclusive_coordinates():
    contig = "N" * 9 + _gene(1500) + "T" * 20
    feats = [RrnaFeature("chr1", 10, 1509, "+", "16S ribosomal RNA")]
    assert extract_16s([("chr1 x", contig)], feats) == _gene(1500)


def test_extract_16s_minus_strand_is_reverse_complemented():
    gene = _gene(1500, "AACGTT")
    contig = "C" * 5 + reverse_complement(gene) + "C" * 5
    feats = [RrnaFeature("chr1", 6, 1505, "-", "16S ribosomal RNA")]
    assert extract_16s([("chr1", contig)], feats) == gene


def test_extract_16s_ignores_other_rrnas_and_partial_16s_and_picks_longest():
    contig = _gene(8000)
    feats = [
        RrnaFeature("chr1", 1, 2900, "+", "23S ribosomal RNA"),
        RrnaFeature("chr1", 3000, 3500, "+", "16S ribosomal RNA"),        # partial: 501 bp
        RrnaFeature("chr1", 4000, 5499, "+", "16S ribosomal RNA"),        # 1500 bp
        RrnaFeature("chr1", 6000, 7541, "+", "16S ribosomal RNA"),        # 1542 bp  <- longest
    ]
    assert len(extract_16s([("chr1", contig)], feats)) == 1542


def test_extract_16s_raises_without_a_full_length_16s():
    feats = [RrnaFeature("chr1", 1, 120, "+", "5S ribosomal RNA")]
    with pytest.raises(DnaSequenceError):
        extract_16s([("chr1", _gene(500))], feats)
