import yaml

from soamp.runs.launch_plan import WANDB_PIN, build_onstart_cmd, build_run_spec, shard, spec_path_for
from soamp.runs.plan import expand
from soamp.runs.request import ExperimentRequest
from soamp.runs.spec import InstanceSpec


def _req(**over):
    base = {
        "experiment": "unit_exp",
        "representations": {"peptide": ["rdkit", "peptideclm"], "organism": ["vocab"]},
        "architectures": [{"name": "baseline_classifier", "hparams": {"weight_decay": [1e-3, 1e-4]}, "epochs": 5}],
    }
    base.update(over)
    return ExperimentRequest(**base)


def test_shard_round_robin_no_empty_shards_and_complete():
    cells = expand(_req())
    parts = shard(cells, 3)
    assert sorted(c.exp_id for p in parts for c in p) == sorted(c.exp_id for c in cells)
    assert len(shard(cells[:1], 4)) == 1


def test_run_spec_lists_cells_prepare_steps_once_and_validates():
    req = _req()
    cells = expand(req)
    spec = build_run_spec(req, cells, InstanceSpec(max_hours=1.0))
    assert len(spec.experiments) == len(cells) == 4
    cmds = [tuple(s.cmd) for s in spec.prepare]
    assert len(cmds) == len(set(cmds))  # vocab step not duplicated across cells
    assert any("peptideclm" in " ".join(c) for c in cmds) and any("organism_vocab" in " ".join(c) for c in cmds)
    # round-trips through the real schema (extra=forbid)
    RunSpec = type(spec)
    assert RunSpec(**yaml.safe_load(yaml.safe_dump(spec.model_dump(mode="json")))).instance.max_hours == 1.0


def test_onstart_pins_wandb_and_keeps_secrets_out():
    cmd = build_onstart_cmd(code_artifact="ent/soamp-ops/code-bundle:v3", data_artifact="ent/soamp-ops/soamp_dev_data:latest",
                            spec_path=str(spec_path_for("unit_exp", 0)), max_hours=1.0, grace_seconds=120)
    assert WANDB_PIN in cmd and "remote_main.sh" in cmd and "config/vast/generated/unit_exp_0.yaml" in cmd
    assert "WANDB_API_KEY" not in cmd and "VAST" not in cmd
    assert "soamp_dev_data:latest" in cmd
