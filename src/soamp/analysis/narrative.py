"""The agent's narrative contract: strict schema + a validator that rejects invented numbers.

The agent writes ``narrative.json`` from ``facts.json`` only. Every number appearing in its
text must exist in the facts (same "every number is computed, never typed" rule as
scripts/slides/build_models_chapter.py); next-experiment drafts must be valid
``ExperimentRequest``s so they can be launched as-is.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from soamp.runs.request import ExperimentRequest

MAX_SLIDES = 6
MAX_BULLETS = 3
SMALL_INTS = range(0, 13)  # counts like "3 runs", "5 folds"
_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w])")


class Slide(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(max_length=80)
    bullets: list[str] = Field(min_length=1, max_length=MAX_BULLETS)
    figure: str | None = None  # one of deck.FIGURES, or None


class NextExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(max_length=80)
    rationale: str
    request: dict  # an ExperimentRequest, validated below

    @field_validator("request")
    @classmethod
    def _is_a_valid_request(cls, v: dict) -> dict:
        ExperimentRequest(**v)
        return v


class Narrative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(max_length=80)
    summary: list[str] = Field(min_length=1, max_length=MAX_BULLETS)
    slides: list[Slide] = Field(default_factory=list, max_length=MAX_SLIDES)
    next_experiments: list[NextExperiment] = Field(default_factory=list, max_length=3)


def _leaves(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _leaves(v)
    else:
        yield x


def allowed_numbers(facts: dict) -> set[float]:
    """Every numeric value in the facts (and numbers embedded in their text), as floats."""
    out: set[float] = set()
    for leaf in _leaves(facts):
        if isinstance(leaf, bool):
            continue
        if isinstance(leaf, (int, float)):
            out.add(float(leaf))
        elif isinstance(leaf, str):
            out.update(float(m) for m in _NUM.findall(leaf))
    return out


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def unknown_numbers(text: str, allowed: set[float]) -> list[str]:
    bad = []
    for tok in _NUM.findall(text):
        v = float(tok)
        d = _decimals(tok)
        if d == 0 and abs(v) in SMALL_INTS:
            continue
        if any(round(a, d) == round(v, d) or round(a, d) == round(-v, d) for a in allowed):
            continue
        bad.append(tok)
    return bad


def _texts(n: Narrative) -> list[tuple[str, str]]:
    out = [("title", n.title)] + [(f"summary[{i}]", s) for i, s in enumerate(n.summary)]
    for i, s in enumerate(n.slides):
        out += [(f"slides[{i}].title", s.title)] + [(f"slides[{i}].bullets[{j}]", b) for j, b in enumerate(s.bullets)]
    for i, e in enumerate(n.next_experiments):
        out += [(f"next_experiments[{i}].title", e.title), (f"next_experiments[{i}].rationale", e.rationale)]
    return out


def validate_narrative(raw: dict, facts: dict, figures: set[str]) -> tuple[Narrative | None, list[str]]:
    """(narrative, problems). Problems are human-readable; the narrative is None if the schema fails."""
    try:
        n = Narrative(**raw)
    except ValidationError as e:
        return None, [f"schema: {err['loc']}: {err['msg']}" for err in e.errors()]
    problems = []
    allowed = allowed_numbers(facts)
    for where, text in _texts(n):
        bad = unknown_numbers(text, allowed)
        if bad:
            problems.append(f"{where}: number(s) {bad} not found in facts.json")
    for i, s in enumerate(n.slides):
        if s.figure is not None and s.figure not in figures:
            problems.append(f"slides[{i}].figure {s.figure!r} is not one of {sorted(figures)}")
    names = {r["name"] for r in facts.get("runs", [])}
    for i, e in enumerate(n.next_experiments):
        if e.request.get("experiment") in names:
            problems.append(f"next_experiments[{i}]: experiment name {e.request['experiment']!r} already exists; use a new name")
    return n, problems
