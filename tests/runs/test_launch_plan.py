import subprocess
import time

import pytest
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


def _cmd():
    return build_onstart_cmd(code_artifact="ent/soamp-ops/code-unit_exp:v3", data_artifact="ent/soamp-ops/soamp_dev_data:latest",
                             spec_path=str(spec_path_for("unit_exp", 0)), max_hours=1.0, grace_seconds=120)


def test_onstart_is_valid_bash_pins_wandb_and_keeps_secrets_out():
    cmd = _cmd()
    assert subprocess.run(["bash", "-n"], input=cmd, text=True).returncode == 0
    assert WANDB_PIN in cmd and "remote_main.sh" in cmd and "config/vast/generated/unit_exp_0.yaml" in cmd
    assert "soamp_dev_data:latest" in cmd and "'/workspace/soamp'" in cmd and "__" not in cmd.replace("__init__", "")
    assert "WANDB_API_KEY" not in cmd


def _run_onstart(tmp_path, curl_code: str, pip_rc: int = 0, py_rc: int = 0):
    """Execute the real script with stub binaries; return (rc, calls, output)."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    calls = tmp_path / "calls.log"
    stubs = {
        "curl": f'echo "curl $*" >> {calls}; printf {curl_code}',
        "pip": f'echo "pip $*" >> {calls}; exit {pip_rc}',
        "python": f'echo "python" >> {calls}; cat > /dev/null; exit {py_rc}',
        "vastai": f'echo "vastai $*" >> {calls}',
        "timeout": 'shift; exec "$@"',
        "setsid": f'echo "setsid $*" >> {calls}',
        "yes": "cat > /dev/null; echo y",
        "cd": None,
    }
    for name, body in stubs.items():
        if body:
            (bin_ / name).write_text(f"#!/bin/bash\n{body}\n")
            (bin_ / name).chmod(0o755)
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "CONTAINER_ID": "123", "CONTAINER_API_KEY": "k"}
    script = _cmd().replace("/workspace/soamp", str(tmp_path)).replace("/workspace/_code", str(tmp_path / "_code")).replace(">/workspace/onstart.log", f">{tmp_path}/onstart.log")
    p = subprocess.run(["/bin/bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)
    time.sleep(0.5)  # the hand-over to remote_main.sh is a background job; let its stub log
    return p.returncode, calls.read_text() if calls.exists() else "", p.stdout + p.stderr


def test_onstart_destroys_instance_immediately_when_wandb_blocks_the_host(tmp_path):
    rc, calls, out = _run_onstart(tmp_path, curl_code="403")
    assert rc == 1 and "FATAL" in out
    assert "vastai destroy instance 123 --api-key k" in calls
    assert "python" not in calls and "setsid" not in calls  # never got as far as the bundle / remote_main


def test_onstart_destroys_instance_when_a_later_step_fails(tmp_path):
    rc, calls, out = _run_onstart(tmp_path, curl_code="404", py_rc=1)
    assert rc == 1 and "code bundle download failed" in out
    assert "vastai destroy instance 123" in calls and "setsid" not in calls


def test_onstart_destroys_instance_when_pip_install_fails(tmp_path):
    rc, calls, out = _run_onstart(tmp_path, curl_code="404", pip_rc=1)
    assert rc == 1 and "pip install wandb failed" in out and "vastai destroy instance 123" in calls


def test_onstart_hands_over_to_remote_main_when_everything_works(tmp_path):
    rc, calls, out = _run_onstart(tmp_path, curl_code="404")
    assert rc == 0 and "handed over" in out
    assert "setsid nohup bash scripts/vast/remote_main.sh" in calls and "vastai destroy" not in calls


def test_data_files_can_be_limited_to_some_peptide_representations():
    from soamp.runs.launch_plan import data_files

    full = data_files("dev_v2_orgbal")
    rdkit = data_files("dev_v2_orgbal", ["rdkit"])
    assert full[:2] == rdkit[:2] and set(rdkit) < set(full)
    assert not any("peptideclm" in f for f in rdkit) and any("rdkit" in f for f in rdkit)
    assert any("organism_vocab" in f for f in rdkit)           # the organism side always ships
    with pytest.raises(ValueError, match="unknown peptide representation"):
        data_files("dev_v2_orgbal", ["nope"])
