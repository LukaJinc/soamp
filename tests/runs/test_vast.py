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
