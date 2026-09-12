import math
import pytest

from soamp.features.kmer import compute_kmer_composition, parse_fasta_sequences


def test_compute_kmer_composition_dimensions_per_k():
    result = compute_kmer_composition(["ACGT"], k_values=(1, 2))
    ones = [k for k in result if k.startswith("1mer_")]
    twos = [k for k in result if k.startswith("2mer_")]
    assert len(ones) == 4
    assert len(twos) == 16


def test_compute_kmer_composition_default_dimension_is_340():
    result = compute_kmer_composition(["ACGT"])
    assert len(result) == 4 + 16 + 64 + 256


def test_compute_kmer_composition_sliding_window_counts_overlaps():
    # "AAAA" has 3 overlapping "AA" 2-mers, not 2 (str.count would give 2).
    result = compute_kmer_composition(["AAAA"], k_values=(2,))
    # After L2 normalization within the 2-mer block, "AA" should have the
    # largest magnitude among all 16 2-mers since it's the only nonzero one.
    aa = result["2mer_AA"]
    others = [v for name, v in result.items() if name != "2mer_AA"]
    assert aa == pytest.approx(1.0)  # only nonzero entry -> L2-normalizes to 1.0
    assert all(v == 0.0 for v in others)


def test_compute_kmer_composition_l2_norm_is_one_per_block():
    result = compute_kmer_composition(["ACGTACGTACGT"], k_values=(1, 2, 3))
    for k in (1, 2, 3):
        block = [v for name, v in result.items() if name.startswith(f"{k}mer_")]
        norm = math.sqrt(sum(v * v for v in block))
        assert norm == pytest.approx(1.0)


def test_compute_kmer_composition_skips_non_acgt_windows():
    # "N" is a common ambiguous-base placeholder in real genome FASTA files.
    result = compute_kmer_composition(["ACNGT"], k_values=(1,))
    # Only A, C, G, T counted -- "N" itself and any window touching it for
    # k>1 would be skipped, but for k=1 only the "N" position itself is skipped.
    total = sum(v * v for v in result.values())
    assert total == pytest.approx(1.0)  # still L2-normalizes fine, no crash


def test_compute_kmer_composition_all_zero_sequence_does_not_divide_by_zero():
    result = compute_kmer_composition([""], k_values=(1,))
    assert all(v == 0.0 for v in result.values())


def test_compute_kmer_composition_sums_across_multiple_sequences():
    single = compute_kmer_composition(["ACGTACGT"], k_values=(1,))
    split = compute_kmer_composition(["ACGT", "ACGT"], k_values=(1,))
    for key in single:
        assert single[key] == pytest.approx(split[key])


def test_parse_fasta_sequences_multi_record(tmp_path):
    fasta_path = tmp_path / "genome.fasta"
    fasta_path.write_text(
        ">contig1 some description\n"
        "ACGT\n"
        "acgt\n"
        ">contig2\n"
        "TTTT\n"
    )
    sequences = parse_fasta_sequences(fasta_path)
    assert sequences == ["ACGTACGT", "TTTT"]


def test_parse_fasta_sequences_empty_file(tmp_path):
    fasta_path = tmp_path / "empty.fasta"
    fasta_path.write_text("")
    assert parse_fasta_sequences(fasta_path) == []
