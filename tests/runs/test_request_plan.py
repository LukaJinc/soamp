import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from soamp.engine.config import TrainConfig
from soamp.runs.plan import expand, request_hash, write_cell_configs
from soamp.runs.request import ExperimentRequest
from soamp.utils.config import REPO_ROOT, load_config


def _req(**over) -> ExperimentRequest:
    base = {
        "experiment": "unit_exp",
        "representations": {"peptide": ["rdkit", "peptideclm"], "organism": ["vocab"]},
        "architectures": [
            {"name": "baseline_classifier", "hparams": {"weight_decay": [1e-3, 1e-4], "hidden_dims": [[32, 16]]}, "epochs": 10},
            {"name": "attention_fusion_classifier", "hparams": {"projection_dim": [32, 64], "learning_rate": 1e-3}},
        ],
    }
    base.update(over)
    return ExperimentRequest(**base)


def test_cv_default_on_and_off_rejected():
    assert _req().cv is True
    with pytest.raises(ValidationError):
        _req(cv=False)


def test_foreign_architecture_key_is_rejected_with_architecture_name():
    with pytest.raises(ValidationError, match="baseline_classifier"):
        _req(architectures=[{"name": "baseline_classifier", "hparams": {"projection_dim": 32}}])


def test_unknown_architecture_dataset_and_representation_rejected():
    with pytest.raises(ValidationError):
        _req(architectures=[{"name": "resnet", "hparams": {}}])
    with pytest.raises(ValidationError):
        _req(dataset="nope")
    with pytest.raises(ValidationError, match="genome2vec"):
        _req(representations={"peptide": ["rdkit"], "organism": ["genome2vec"]})


def test_bad_experiment_slug_rejected():
    with pytest.raises(ValidationError):
        _req(experiment="Bad Name")


def test_grid_is_crossed_per_architecture_not_across_them():
    cells = expand(_req())
    by_arch: dict[str, int] = {}
    for c in cells:
        by_arch[c.arch] = by_arch.get(c.arch, 0) + 1
    # mlp: 2 wd x 1 hidden x 2 peptide = 4; attn: 2 proj x 2 peptide = 4
    assert by_arch == {"baseline_classifier": 4, "attention_fusion_classifier": 4}
    # an MLP cell never carries attention keys and vice versa
    for c in cells:
        assert set(c.config["model"]) - {"architecture"} <= {c.arch}


def test_hidden_dims_flat_list_is_one_value_list_of_lists_is_axis():
    one = expand(_req(architectures=[{"name": "baseline_classifier", "hparams": {"hidden_dims": [32, 16]}}]))
    two = expand(_req(architectures=[{"name": "baseline_classifier", "hparams": {"hidden_dims": [[32, 16], [64]]}}]))
    assert len(one) == 2 and len(two) == 4  # x2 peptide representations


def test_expansion_is_deterministic_and_names_unique():
    a, b = expand(_req()), expand(_req())
    assert [c.exp_id for c in a] == [c.exp_id for c in b]
    assert len({c.exp_id for c in a}) == len(a)
    assert request_hash(_req()) == request_hash(_req())
    assert request_hash(_req()) != request_hash(_req(seeds=[1]))


def test_seeds_multiply_cells():
    assert len(expand(_req(seeds=[1, 2, 3]))) == 3 * len(expand(_req()))


def test_random_sampling_is_seeded_and_capped():
    r = _req(sampling={"mode": "random", "n": 3})
    assert len(expand(r)) == 3 and [c.exp_id for c in expand(r)] == [c.exp_id for c in expand(r)]
    with pytest.raises(ValidationError):
        _req(sampling={"mode": "random"})


def test_cells_carry_group_tags_and_inputs():
    c = expand(_req())[0]
    assert c.config["wandb_group"] == "unit_exp"
    assert "cv" in c.config["wandb_tags"] and any(t.startswith("req_") for t in c.config["wandb_tags"])
    assert c.config["input_files"]["organism_vocab_filename"] == "organism_vocab_vocab_embedding.json"


def test_written_configs_validate_against_train_config(tmp_path: Path):
    shutil.copytree(REPO_ROOT / "config/train", tmp_path / "config/train", ignore=shutil.ignore_patterns("generated"))
    req = _req()
    cells = expand(req)
    paths = write_cell_configs(req, cells, repo_root=tmp_path)
    assert len(paths) == len(cells)
    for p, cell in zip(paths, cells):
        cfg = load_config(tmp_path / p, TrainConfig)
        assert cfg.exp_id == cell.exp_id
        assert cfg.wandb_group == "unit_exp"
        assert cfg.model.architecture == cell.arch
        assert cfg.paths.data_dir.name == "dev_v2"  # the default dataset's base config


def test_tree_architecture_has_its_own_hparams_and_no_epochs_or_optimiser_fields():
    req = _req(architectures=[{"name": "hist_gradient_boosting_classifier",
                               "hparams": {"max_leaf_nodes": [15, 31], "learning_rate": [0.05, 0.1], "max_iter": 100}}])
    cells = expand(req)
    assert len(cells) == 2 * 2 * 2  # leaves x lr x 2 peptide representations
    body = cells[0].config
    assert body["model"]["architecture"] == "hist_gradient_boosting_classifier"
    assert set(body["model"]["hist_gradient_boosting_classifier"]) == {"max_iter", "learning_rate", "max_leaf_nodes"}
    assert "optim" not in body  # the tree's learning_rate is NOT the optimiser's
    assert "epochs" not in body["loop"]
    assert cells[0].exp_id.split("__")[1] == "hgb"
    with pytest.raises(ValidationError, match="max_iter"):
        _req(architectures=[{"name": "hist_gradient_boosting_classifier", "epochs": 10}])
    with pytest.raises(ValidationError, match="hist_gradient_boosting_classifier"):
        _req(architectures=[{"name": "hist_gradient_boosting_classifier", "hparams": {"projection_dim": 8}}])


def test_tree_architecture_rejected_with_molgraph():
    with pytest.raises(ValidationError, match="molgraph"):
        _req(representations={"peptide": ["molgraph"], "organism": ["vocab"]},
             architectures=[{"name": "hist_gradient_boosting_classifier"}])
    assert expand(_req(representations={"peptide": ["molgraph"], "organism": ["vocab"]},
                       architectures=[{"name": "baseline_classifier"}]))


def test_tree_cell_config_validates_against_train_config(tmp_path):
    shutil.copytree(REPO_ROOT / "config/train", tmp_path / "config/train", ignore=shutil.ignore_patterns("generated"))
    req = _req(architectures=[{"name": "hist_gradient_boosting_classifier", "hparams": {"max_iter": 50, "learning_rate": 0.05}}])
    cells = expand(req)
    for p, cell in zip(write_cell_configs(req, cells, repo_root=tmp_path), cells):
        cfg = load_config(tmp_path / p, TrainConfig)
        assert cfg.model.architecture == "hist_gradient_boosting_classifier"
        assert cfg.model.hist_gradient_boosting_classifier.max_iter == 50
        assert cfg.model.hist_gradient_boosting_classifier.learning_rate == 0.05


# ---- dataset default + the dev_v2 registry entry ---------------------------------------------

def test_default_dataset_is_the_seven_strain_dev_v2_and_dev_stays_selectable():
    assert _req().dataset == "dev_v2"
    assert _req(dataset="dev").dataset == "dev"
    assert ExperimentRequest.model_fields["dataset"].default == "dev_v2"


@pytest.mark.parametrize("dataset,folder", [("dev", "dev"), ("dev_v2", "dev_v2")])
def test_cells_of_each_dataset_inherit_their_own_data_dir(tmp_path, dataset, folder):
    shutil.copytree(REPO_ROOT / "config/train", tmp_path / "config/train", ignore=shutil.ignore_patterns("generated"))
    req = _req(dataset=dataset)
    cells = expand(req)
    for p in write_cell_configs(req, cells, repo_root=tmp_path):
        cfg = load_config(tmp_path / p, TrainConfig)
        assert cfg.paths.data_dir.name == folder
    assert all(f"{dataset}_dataset" in c.config["wandb_tags"] for c in cells)


def test_registry_entries_are_self_consistent():
    from soamp.runs.launch_plan import data_files
    from soamp.runs.request import DATASETS

    for name, ds in DATASETS.items():
        folder = f"data/{name}/"
        assert all(f.startswith(folder) for f in ds["required_artifacts"]), name
        files = data_files(name)
        assert files[: len(ds["required_artifacts"])] == ds["required_artifacts"]
        assert all(f.startswith(folder) for f in files), f"{name} ships a file from another dataset"
        for group, kinds in ds["prepare"].items():           # every prepare step builds a listed representation file
            for kind, steps in kinds.items():
                listed = ds[group][kind]
                for _cmd, produces in steps:
                    assert all(p.rsplit("/", 1)[1] in listed for p in produces), (name, kind, produces)
                    assert all(p.startswith(folder) for p in produces)
        for _cmd, _prod in (s for kinds in ds["prepare"].values() for steps in kinds.values() for s in steps):
            cfg = next(a for a in _cmd if a.startswith("config/features/"))
            assert (REPO_ROOT / cfg).is_file(), f"{name}: missing feature config {cfg}"
        assert (REPO_ROOT / "config/train" / ds["base_config"]).is_file()


def test_dev_v2_data_artifact_lists_every_cached_feature_file():
    from soamp.runs.launch_plan import data_files

    assert data_files("dev_v2") == [
        "data/dev_v2/mic_classification_dataset.csv", "data/dev_v2/peptide_split.csv",
        "data/dev_v2/peptide_features_rdkit.csv", "data/dev_v2/peptide_feature_scaler_rdkit.json",
        "data/dev_v2/peptide_features_peptideclm.csv", "data/dev_v2/peptide_feature_scaler_peptideclm.json",
        "data/dev_v2/peptide_feature_scaler_molgraph.json", "data/dev_v2/organism_vocab_vocab_embedding.json",
        "data/dev_v2/organism_vocab_kmer_composition.json", "data/dev_v2/organism_vocab_dnabert_s_16s.json",
    ]


# ---- XGBoost architecture ---------------------------------------------------------------------

def test_xgboost_architecture_has_its_own_hparams_and_no_epochs_or_optimiser_fields():
    req = _req(architectures=[{"name": "xgboost_classifier",
                               "hparams": {"max_depth": [4, 6], "learning_rate": [0.05, 0.1], "n_estimators": 100}}])
    cells = expand(req)
    assert len(cells) == 2 * 2 * 2  # depth x lr x 2 peptide representations
    body = cells[0].config
    assert body["model"]["architecture"] == "xgboost_classifier"
    assert set(body["model"]["xgboost_classifier"]) == {"n_estimators", "learning_rate", "max_depth"}
    assert "optim" not in body and "epochs" not in body["loop"]
    assert cells[0].exp_id.split("__")[1] == "xgb"
    with pytest.raises(ValidationError, match="n_estimators"):
        _req(architectures=[{"name": "xgboost_classifier", "epochs": 10}])
    with pytest.raises(ValidationError, match="xgboost_classifier"):
        _req(architectures=[{"name": "xgboost_classifier", "hparams": {"max_iter": 8}}])      # an HGB-only key
    with pytest.raises(ValidationError, match="hist_gradient_boosting_classifier"):
        _req(architectures=[{"name": "hist_gradient_boosting_classifier", "hparams": {"max_depth": 3}}])  # and back


def test_xgboost_rejected_with_molgraph_and_cell_config_validates(tmp_path):
    with pytest.raises(ValidationError, match="molgraph"):
        _req(representations={"peptide": ["molgraph"], "organism": ["vocab"]}, architectures=[{"name": "xgboost_classifier"}])
    shutil.copytree(REPO_ROOT / "config/train", tmp_path / "config/train", ignore=shutil.ignore_patterns("generated"))
    req = _req(architectures=[{"name": "xgboost_classifier", "hparams": {"n_estimators": 50, "max_depth": 4, "subsample": 0.8}}])
    cells = expand(req)
    for p in write_cell_configs(req, cells, repo_root=tmp_path):
        cfg = load_config(tmp_path / p, TrainConfig)
        assert cfg.model.architecture == "xgboost_classifier"
        assert (cfg.model.xgboost_classifier.n_estimators, cfg.model.xgboost_classifier.max_depth,
                cfg.model.xgboost_classifier.subsample) == (50, 4, 0.8)
