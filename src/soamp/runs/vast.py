"""Thin, mockable wrapper over the `vastai` CLI.

All side effects go through one seam, ``Runner`` (args -> stdout JSON text), so
tests inject a fake and nothing here touches the network on import. The real
runner shells out to ``vastai <args> --raw`` (the CLI reads VAST_API_KEY from
`vastai set api-key`, which the cloud setup script / launcher does once).

Offers are ranked by host reliability, not price: a dropped host loses the
in-flight cell (there is no resume). Every instance created here carries the
label ``soamp-<experiment>`` so the janitor and the concurrency cap can find it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable

import yaml

from soamp.runs.spec import InstanceSpec, offer_query
from soamp.utils.config import REPO_ROOT

LABEL_PREFIX = "soamp-"
Runner = Callable[[list[str]], str]


class VastError(RuntimeError):
    pass


def cli_runner(binary: str | None = None) -> Runner:
    exe = binary or shutil.which("vastai")
    if exe is None:
        raise VastError("`vastai` CLI not found (pip install vastai)")

    key = os.environ.get("VAST_API_KEY")
    auth = ["--api-key", key] if key else []  # per-call, so nothing is written to ~/.config

    def run(args: list[str]) -> str:
        # `vastai destroy` asks "[y/N]" and aborts without a tty (seen live); answering "y" is
        # harmless for every other command, which never reads stdin.
        p = subprocess.run([exe, *args, *auth, "--raw"], capture_output=True, text=True, input="y\n")
        if p.returncode != 0:
            raise VastError(f"vastai {' '.join(args[:2])} failed rc={p.returncode}: {(p.stderr or p.stdout).strip()[:500]}")
        return p.stdout

    return run


@dataclass(frozen=True)
class Offer:
    id: int
    gpu_name: str
    price_per_hour: float
    reliability: float
    inet_down_mbps: float
    location: str
    host_id: int = 0
    machine_id: int = 0


@dataclass(frozen=True)
class VastInstance:
    id: int
    label: str
    status: str  # vast `actual_status`: loading / running / exited ...
    price_per_hour: float
    started_at: float | None = None  # unix seconds (vast `start_date`), None if unknown


def _json(text: str, what: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise VastError(f"could not parse `{what}` output as JSON: {text[:200]!r}") from e


BLOCKED_HOSTS_PATH = "config/runs/blocked_hosts.yaml"


def load_blocked_host_ids(path: str = BLOCKED_HOSTS_PATH) -> frozenset[int]:
    f = REPO_ROOT / path
    return frozenset(int(h) for h in (yaml.safe_load(f.read_text()) or {}).get("blocked_host_ids", [])) if f.is_file() else frozenset()


class VastClient:
    def __init__(self, runner: Runner | None = None, blocked_host_ids: frozenset[int] | set[int] = frozenset()):
        self._run = runner or cli_runner()
        self._blocked = frozenset(blocked_host_ids)

    def search_offers(self, instance: InstanceSpec, limit: int = 8) -> list[Offer]:
        """Most reliable first; price only breaks ties."""
        raw = _json(self._run(["search", "offers", offer_query(instance), "-o", "reliability-"]), "search offers")
        offers = [
            Offer(
                id=int(o["id"]), gpu_name=o["gpu_name"], price_per_hour=float(o["dph_total"]),
                reliability=float(o["reliability2"]), inet_down_mbps=float(o["inet_down"]),
                location=str(o.get("geolocation", "")), host_id=int(o.get("host_id") or 0), machine_id=int(o.get("machine_id") or 0),
            )
            for o in raw
            if int(o.get("host_id") or 0) not in self._blocked
        ]
        offers.sort(key=lambda o: (-o.reliability, o.price_per_hour))
        return offers[:limit]

    def create_instance(
        self, offer_id: int, *, image: str, disk_gb: int, label: str, onstart_cmd: str, env: dict[str, str]
    ) -> int:
        """Rent an instance that bootstraps itself (no SSH from the caller)."""
        if not label.startswith(LABEL_PREFIX):
            raise ValueError(f"label must start with {LABEL_PREFIX!r} so the janitor can find it")
        env_flags = " ".join(f"-e {k}={v}" for k, v in env.items())
        out = _json(
            self._run([
                "create", "instance", str(offer_id), "--image", image, "--disk", str(disk_gb),
                "--label", label, "--ssh", "--direct", "--env", env_flags, "--onstart-cmd", onstart_cmd,
            ]),
            "create instance",
        )
        if "new_contract" not in out:
            raise VastError(f"create instance returned no new_contract: {out}")
        return int(out["new_contract"])

    def list_instances(self, label_prefix: str = LABEL_PREFIX) -> list[VastInstance]:
        raw = _json(self._run(["show", "instances"]), "show instances")
        return [
            VastInstance(
                id=int(i["id"]), label=str(i.get("label") or ""), status=str(i.get("actual_status") or ""),
                price_per_hour=float(i.get("dph_total") or 0.0),
                started_at=float(i["start_date"]) if i.get("start_date") else None,
            )
            for i in raw
            if str(i.get("label") or "").startswith(label_prefix)
        ]

    def destroy_instance(self, instance_id: int) -> None:
        self._run(["destroy", "instance", str(instance_id)])
