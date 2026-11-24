import hashlib
import subprocess

from soamp.runs.data_artifact import build_metadata, dataset_counts, describe, git_state, sha256_of, verify_download

CSV = ("peptide_id,sequence,smiles,organism,ncbi_taxon_id_if_available,mic_value_uM,mic_type,has_noncanonical,label,split\n"
       "1,AK,CC,E. coli,,5,exact,False,active,train\n"
       "1,AK,CC,S. aureus,,500,exact,False,inactive,train\n"
       "2,GK,CC,E. coli,,7,exact,True,active,test\n")


def _repo(tmp_path):
    (tmp_path / "data/dev_v2").mkdir(parents=True)
    (tmp_path / "data/dev_v2/mic_classification_dataset.csv").write_text(CSV)
    (tmp_path / "data/dev_v2/features.csv").write_text("a,b\n1,2\n")
    return tmp_path


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"hello" * 100000)
    assert sha256_of(p) == hashlib.sha256(b"hello" * 100000).hexdigest()


def test_dataset_counts(tmp_path):
    c = dataset_counts(_repo(tmp_path) / "data/dev_v2/mic_classification_dataset.csv")
    assert c == {"labelled_pairs": 3, "peptides": 2, "labels": {"active": 2, "inactive": 1},
                 "split_rows": {"test": 1, "train": 2}, "pairs_per_organism": {"E. coli": 2, "S. aureus": 1}}


def test_metadata_describe_and_verify_roundtrip(tmp_path):
    root = _repo(tmp_path)
    files = ["data/dev_v2/mic_classification_dataset.csv", "data/dev_v2/features.csv"]
    meta = build_metadata("dev_v2", root, files, files[0])
    assert set(meta["files"]) == set(files) and meta["counts"]["peptides"] == 2
    assert "3 labelled pairs" in describe(meta) and "2 strains" in describe(meta)
    assert verify_download(meta, root) == []
    (root / files[1]).write_text("a,b\n1,3\n")                       # same size, different content
    assert verify_download(meta, root) == [f"sha256 differs: {files[1]}"]
    (root / files[1]).write_text("longer content")
    assert verify_download(meta, root) == [f"size differs: {files[1]}"]
    (root / files[1]).unlink()
    assert verify_download(meta, root) == [f"missing: {files[1]}"]


def test_git_state_in_and_outside_a_repo(tmp_path):
    assert git_state(tmp_path) == {"sha": None, "dirty": None}        # not a repository
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"], cwd=tmp_path, check=True)
    state = git_state(tmp_path)
    assert len(state["sha"]) == 40 and state["dirty"] is False
