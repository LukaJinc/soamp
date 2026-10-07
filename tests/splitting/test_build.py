import pytest

from soamp.splitting.build import (
    SplitBuildError,
    build_peptide_split,
    unique_peptides_for_splitting,
)


def _row(peptide_id, organism, sequence, smiles, has_noncanonical, label="active"):
    return {
        "label": label,
        "peptide_id": peptide_id,
        "organism": organism,
        "sequence": sequence,
        "smiles": smiles,
        "has_noncanonical": "True" if has_noncanonical else "False",
    }


def test_unique_peptides_for_splitting_dedups_across_organisms():
    rows = [
        _row("1", "Escherichia coli", "AAAAAAAAAA", "CCO", False),
        _row("1", "Staphylococcus aureus", "AAAAAAAAAA", "CCO", False),
        _row("2", "Escherichia coli", "KKKKKKKKKK", "CCN", True),
    ]
    peptides = unique_peptides_for_splitting(rows)
    assert len(peptides) == 2
    assert {p["peptide_id"] for p in peptides} == {"1", "2"}
    p2 = next(p for p in peptides if p["peptide_id"] == "2")
    assert p2["has_noncanonical"] is True
    p1 = next(p for p in peptides if p["peptide_id"] == "1")
    assert (p1["n_rows"], p1["n_active"]) == (2, 2)
    assert (p2["n_rows"], p2["n_active"]) == (1, 1)


def test_unique_peptides_for_splitting_counts_records_and_active():
    rows = [
        _row("1", "E", "AAAAAAAAAA", "CCO", False, label="active"),
        _row("1", "S", "AAAAAAAAAA", "CCO", False, label="inactive"),
        _row("1", "P", "AAAAAAAAAA", "CCO", False, label="active"),
    ]
    (p,) = unique_peptides_for_splitting(rows)
    assert (p["n_rows"], p["n_active"]) == (3, 2)


def test_unique_peptides_for_splitting_raises_on_conflict():
    rows = [
        _row("1", "Escherichia coli", "AAAAAAAAAA", "CCO", False),
        _row("1", "Staphylococcus aureus", "AAAAAAAAAC", "CCO", False),  # conflicting sequence
    ]
    with pytest.raises(SplitBuildError):
        unique_peptides_for_splitting(rows)


_TEST_PEPTIDES = [
    # Near-duplicate SMILES pair (identical aspirin SMILES).
    {"peptide_id": "1", "sequence": "AAAAAAAAAA", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "has_noncanonical": False},
    {"peptide_id": "2", "sequence": "KKKKKKKKKK", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "has_noncanonical": False},
    # Near-duplicate sequence pair (single substitution), unrelated SMILES.
    {"peptide_id": "3", "sequence": "WWWWWWWWWW", "smiles": "CCCCCCCCCCCCCCCC", "has_noncanonical": False},
    {"peptide_id": "4", "sequence": "WWWWWWWWWY", "smiles": "CCCCCCCCCCCCCCCCC", "has_noncanonical": True},
    # X-placeholder peptide: can only connect via fingerprint, never QMAP.
    {"peptide_id": "5", "sequence": "AAAAAAAAXA", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "has_noncanonical": True},
    # Isolated singletons, no edges to anything.
    {"peptide_id": "6", "sequence": "MPRTQSILVK", "smiles": "c1ccc2ccccc2c1", "has_noncanonical": False},
    {"peptide_id": "7", "sequence": "GDCVEFHNLQ", "smiles": "O=C1CCCCC1", "has_noncanonical": False},
]


def test_build_peptide_split_end_to_end():
    result = build_peptide_split(
        _TEST_PEPTIDES,
        fingerprint_threshold=0.8,
        identity_threshold=0.6,
        identity_show_progress=False,
        test_size=0.3,
        n_folds=2,
        bucket_n_iterations=500,
    )
    peptide_ids = {p["peptide_id"] for p in _TEST_PEPTIDES}
    result_ids = {r["peptide_id"] for r in result.rows}
    assert result_ids == peptide_ids

    by_id = {r["peptide_id"]: r for r in result.rows}
    assert by_id["1"]["community"] == by_id["2"]["community"]  # near-dup SMILES co-locate
    assert by_id["3"]["community"] == by_id["4"]["community"]  # near-dup sequence co-locate
    assert by_id["5"]["split"] in ("train", "test")  # X-placeholder still gets a valid split

    for r in result.rows:
        if r["split"] == "test":
            assert r["fold_id"] is None
        else:
            assert r["fold_id"] in (0, 1)

    sidecar = result.sidecar
    for key in (
        "method", "n_peptides", "fingerprint_threshold", "identity_threshold",
        "n_clusters", "largest_cluster_fraction", "dataset_has_noncanonical_fraction",
        "n_qmap_unscoreable_peptides", "bucket_stats",
    ):
        assert key in sidecar
    assert set(sidecar["bucket_stats"].keys()) == {"test", "fold_0", "fold_1"}
    assert sidecar["n_qmap_unscoreable_peptides"] == 1  # only peptide "5"


def test_build_peptide_split_is_deterministic():
    kwargs = dict(test_size=0.3, n_folds=2, bucket_n_iterations=500, identity_show_progress=False)
    r1 = build_peptide_split(_TEST_PEPTIDES, **kwargs)
    r2 = build_peptide_split(_TEST_PEPTIDES, **kwargs)
    assert r1.rows == r2.rows


def test_build_peptide_split_never_separates_hard_linked_peptides():
    # Many peptides with identical SMILES to peptide "0" must share its bucket.
    peptides = [
        {"peptide_id": str(i), "sequence": "ACDEFGHIKL"[: 10] if i else "ACDEFGHIKL",
         "smiles": "CC(=O)Oc1ccccc1C(=O)O", "has_noncanonical": False}
        for i in range(6)
    ] + [
        {"peptide_id": f"u{i}", "sequence": "MPRTQSILVK"[:9] + "ACDEFGHIKLMNPQRSTVWY"[i],
         "smiles": "C" * (i + 3), "has_noncanonical": i % 2 == 0}
        for i in range(20)
    ]
    for p in peptides:
        p["n_rows"], p["n_active"] = 2, 1
    result = build_peptide_split(
        peptides, test_size=0.3, n_folds=2, bucket_n_iterations=500, identity_show_progress=False
    )
    by_id = {r["peptide_id"]: (r["split"], r["fold_id"]) for r in result.rows}
    assert len({by_id[str(i)] for i in range(6)}) == 1
    assert result.sidecar["cross_bucket_edges"]["hard_links"] == 0
