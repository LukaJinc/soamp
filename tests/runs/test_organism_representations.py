"""The launcher accepts every organism representation (vocab, k-mer, DNABERT-S), and refuses
a prebuilt one (k-mer, DNABERT-S) whose artifact is missing or leaves an organism unresolved."""
import json
import shutil

import pytest

from soamp.engine.config import TrainConfig
from soamp.runs.coverage import coverage_problems, dataset_organisms, unresolved_organisms
from soamp.runs.launch_plan import build_run_spec, data_files, prebuilt_files
from soamp.runs.plan import expand, write_cell_configs
from soamp.runs.request import DATASETS, ORGANISM_REPRESENTATIONS, ExperimentRequest
from soamp.runs.spec import InstanceSpec
from soamp.utils.config import REPO_ROOT, load_config

ALL = ["vocab", "kmer", "dnabert"]


def _req(dataset="dev", organism=ALL, **over) -> ExperimentRequest:
    body = {
        "experiment": "org_repr_unit",
        "dataset": dataset,
        "representations": {"peptide": ["rdkit"], "organism": list(organism)},
        "architectures": [{"name": "baseline_classifier", "epochs": 2}],
    }
    body.update(over)
    return ExperimentRequest(**body)


@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_every_dataset_accepts_all_three_organism_representations(dataset):
    assert set(DATASETS[dataset]["organism"]) == set(ALL)
    cells = expand(_req(dataset))
    assert sorted(c.organism for c in cells) == sorted(ALL)
    for c in cells:
        filename, method = ORGANISM_REPRESENTATIONS[c.organism]
        assert c.config["input_files"]["organism_vocab_filename"] == filename
        assert method in c.config["wandb_tags"]
        assert f"_{c.organism}__" in c.exp_id


def test_unknown_organism_representation_lists_the_available_ones():
    with pytest.raises(ValueError, match=r"\['dnabert', 'kmer', 'vocab'\]"):
        _req(organism=["genome2vec"])


@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_generated_kmer_and_dnabert_cells_validate_as_train_configs(tmp_path, dataset):
    shutil.copytree(REPO_ROOT / "config/train", tmp_path / "config/train", ignore=shutil.ignore_patterns("generated"))
    req = _req(dataset, organism=["kmer", "dnabert"])
    for p in write_cell_configs(req, expand(req), repo_root=tmp_path):
        cfg = load_config(tmp_path / p, TrainConfig)
        assert cfg.input_files.organism_vocab_filename in {"organism_vocab_kmer_composition.json", "organism_vocab_dnabert_s_16s.json"}


def test_prebuilt_artifacts_are_required_on_the_instance_and_never_built_there():
    req = _req("dev")
    spec = build_run_spec(req, expand(req), InstanceSpec())
    required = [str(p) for p in spec.required_artifacts]
    assert "data/dev/organism_vocab_kmer_composition.json" in required
    assert "data/dev/organism_vocab_dnabert_s_16s.json" in required
    assert not any("organism_kmer" in " ".join(s.cmd) or "organism_dnabert" in " ".join(s.cmd) for s in spec.prepare)
    vocab_only = _req("dev", organism=["vocab"])
    spec = build_run_spec(vocab_only, expand(vocab_only), InstanceSpec())
    assert [str(p) for p in spec.required_artifacts] == DATASETS["dev"]["required_artifacts"]   # unchanged for vocab
    assert prebuilt_files("dev", ["vocab"]) == []


@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_data_artifact_ships_the_prebuilt_organism_files(dataset):
    files = data_files(dataset)
    assert f"data/{dataset}/organism_vocab_kmer_composition.json" in files
    assert f"data/{dataset}/organism_vocab_dnabert_s_16s.json" in files


@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_prebuilt_feature_configs_exist_and_point_at_the_dataset(dataset):
    from soamp.features.config import FeaturesConfig

    for rep in ("kmer", "dnabert"):
        cfg = load_config(REPO_ROOT / f"config/features/{dataset}_organism_{rep}.yaml", FeaturesConfig)
        assert cfg.paths.data_dir.as_posix().endswith(f"data/{dataset}")
        assert cfg.output_files.organism_vocab_filename == ORGANISM_REPRESENTATIONS[rep][0]
        assert cfg.organism_featurization.method == ORGANISM_REPRESENTATIONS[rep][1]


def test_checked_in_dev_artifacts_cover_every_dev_strain_with_nonzero_vectors():
    assert coverage_problems(_req("dev")) == []
    organisms = dataset_organisms("dev")
    for rep in ("kmer", "dnabert"):
        path = REPO_ROOT / f"data/dev/{ORGANISM_REPRESENTATIONS[rep][0]}"
        assert unresolved_organisms(path, organisms) == []


def test_dev_v2_kmer_and_dnabert_are_refused_until_their_artifacts_exist():
    problems = coverage_problems(_req("dev_v2"))
    if not problems:
        pytest.skip("dev_v2 k-mer/DNABERT-S artifacts have been built: nothing left to refuse")
    assert all("Build it on the laptop" in p or "To fix" in p for p in problems)
    assert coverage_problems(_req("dev_v2", organism=["vocab"])) == []


def _fake_repo(tmp_path, organisms, kmer_species):
    (tmp_path / "data/dev").mkdir(parents=True)
    rows = "peptide_id,organism\n" + "".join(f"{i},{o}\n" for i, o in enumerate(organisms))
    (tmp_path / "data/dev/mic_classification_dataset.csv").write_text(rows)
    artifact = {"method": "kmer_composition", "k_values": [1], "feature_dim": 4,
                "species_features": {s: [0.25] * 4 for s in kmer_species}, "genus_features": {}}
    (tmp_path / "data/dev/organism_vocab_kmer_composition.json").write_text(json.dumps(artifact))
    return tmp_path


def test_incomplete_prebuilt_artifact_names_the_organisms_that_would_be_zero(tmp_path):
    root = _fake_repo(tmp_path, ["Escherichia coli ATCC 25922", "Klebsiella pneumoniae ATCC 700603"], ["Escherichia coli"])
    (problem,) = coverage_problems(_req("dev", organism=["kmer"]), repo_root=root)
    assert "Klebsiella pneumoniae ATCC 700603" in problem and "Escherichia coli ATCC 25922" not in problem
    assert "all-zero" in problem and "dev_organism_kmer.yaml" in problem


def test_missing_prebuilt_artifact_is_refused_with_build_steps(tmp_path):
    root = _fake_repo(tmp_path, ["Escherichia coli ATCC 25922"], ["Escherichia coli"])
    (problem,) = coverage_problems(_req("dev", organism=["dnabert"]), repo_root=root)
    assert "organism_vocab_dnabert_s_16s.json does not exist" in problem
    assert "04_embed_organisms.ipynb" in problem
