import json

import pytest

from soamp.runs.spec import InstanceSpec
from soamp.runs.vast import VastClient, VastError


class FakeRunner:
    def __init__(self, responses: dict[str, str]):
        self.responses, self.calls = responses, []

    def __call__(self, args: list[str]) -> str:
        self.calls.append(args)
        return self.responses[" ".join(args[:2])]


def _offer(i, rel, price):
    return {"id": i, "gpu_name": "RTX_4090", "dph_total": price, "reliability2": rel, "inet_down": 900, "geolocation": "DE"}


def test_offers_ranked_by_reliability_then_price():
    r = FakeRunner({"search offers": json.dumps([_offer(1, 0.990, 0.2), _offer(2, 0.999, 0.9), _offer(3, 0.999, 0.4)])})
    got = VastClient(r).search_offers(InstanceSpec(), limit=2)
    assert [o.id for o in got] == [3, 2]
    assert "-o" in r.calls[0] and "reliability-" in r.calls[0]


def test_create_returns_contract_and_passes_label_onstart_env():
    r = FakeRunner({"create instance": json.dumps({"success": True, "new_contract": 777})})
    iid = VastClient(r).create_instance(5, image="img", disk_gb=30, label="soamp-x", onstart_cmd="echo hi", env={"A": "1", "B": "2"})
    assert iid == 777
    args = r.calls[0]
    assert args[args.index("--label") + 1] == "soamp-x"
    assert args[args.index("--onstart-cmd") + 1] == "echo hi"
    assert args[args.index("--env") + 1] == "-e A=1 -e B=2"


def test_create_requires_soamp_label_and_contract():
    c = VastClient(FakeRunner({"create instance": json.dumps({"success": False})}))
    with pytest.raises(ValueError):
        c.create_instance(5, image="i", disk_gb=1, label="other", onstart_cmd="", env={})
    with pytest.raises(VastError):
        c.create_instance(5, image="i", disk_gb=1, label="soamp-x", onstart_cmd="", env={})


def test_list_instances_filters_to_soamp_labels():
    raw = [{"id": 1, "label": "soamp-a", "actual_status": "running", "dph_total": 0.3},
           {"id": 2, "label": "someone-else", "actual_status": "running", "dph_total": 1.0},
           {"id": 3, "label": None, "actual_status": "exited", "dph_total": 0.1}]
    got = VastClient(FakeRunner({"show instances": json.dumps(raw)})).list_instances()
    assert [(i.id, i.status) for i in got] == [(1, "running")]


def test_bad_json_raises_vast_error():
    with pytest.raises(VastError):
        VastClient(FakeRunner({"show instances": "not json"})).list_instances()


def test_blocked_hosts_are_never_offered():
    raw = [dict(_offer(1, 0.999, 0.3), host_id=46431, machine_id=51069), dict(_offer(2, 0.99, 0.5), host_id=7, machine_id=8)]
    r = FakeRunner({"search offers": json.dumps(raw)})
    assert [o.id for o in VastClient(r, blocked_host_ids={46431}).search_offers(InstanceSpec())] == [2]
    assert [o.id for o in VastClient(r).search_offers(InstanceSpec())] == [1, 2]


def test_committed_blocklist_contains_the_known_bad_host():
    from soamp.runs.vast import load_blocked_host_ids

    assert 46431 in load_blocked_host_ids()


def test_cli_runner_answers_the_destroy_confirmation(tmp_path, monkeypatch):
    import subprocess
    from soamp.runs.vast import cli_runner

    seen = {}
    def fake_run(cmd, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    cli_runner("/bin/echo")(["destroy", "instance", "1"])
    assert seen["input"] == "y\n"  # no tty in cloud/cron: the [y/N] prompt must be pre-answered
