"""Shared fixtures: a fake vision model that answers from the ground truth, and cached synthetic cases."""
from __future__ import annotations

from functools import cache

import pytest

from cailens import form, llm, synth
from cailens.truth import expected


@cache
def case(seed: int, split: str = "dev"):
    return synth.generate(seed, split)


class FakeModel:
    """Answers like a perfect transcriber (optionally with planted mistakes), and records every call."""

    def __init__(self) -> None:
        self.truths: dict[str, synth.CaseTruth] = {}
        self.calls: list[str] = []
        self.mistakes: dict[str, str | None] = {}  # field -> value to answer instead of the truth
        self.illegible: set[str] = set()
        self.fail = False
        self.verifier_down = False
        self.second_reading: dict[str, str] = {}  # field -> what the verifier reads instead of the truth

    def __call__(self, *, purpose: str, cache_id: str | None = None, **kw):
        self.calls.append(purpose)
        if self.fail:
            raise llm.LLMUnavailable("finto errore di rete")
        t = self.truths.get((cache_id or "").rsplit(":", 1)[0])
        if t is None:  # a case the fake does not know behaves like a model that is not there
            raise llm.LLMUnavailable("caso sconosciuto al modello finto")
        exp = expected(t)
        if purpose == "handwriting":
            ans = {f: self.mistakes.get(f, exp[f]) for f in form.TEXT_FIELDS}
            ans["illeggibili"] = sorted(self.illegible)
        elif purpose == "verifier":
            if self.verifier_down:
                raise llm.LLMUnavailable("verificatore giù")
            ans = {f: exp[f] or "" for f in ("data", "A.targa", "B.targa", "A.polizza", "B.polizza", "A.cf", "B.cf")}
            ans.update(self.second_reading)
        elif purpose == "sketch":
            ans = {"scenario": t.scenario, "impatto_A": t.A.impatto, "impatto_B": t.B.impatto, "certezza": "alta",
                   "descrizione": "schizzo"}
        else:  # baseline: reads text well, misses one cross per side
            ans = {"caselle_A": t.A.circostanze[1:], "caselle_B": t.B.circostanze, "caselle_incerte": [],
                   "feriti": "si" if t.feriti else "no", "veicoli": "no", "oggetti": "no",
                   "firma_A": t.A.firma, "firma_B": True}
            ans.update({f.replace(".", "_"): exp[f] for f in form.TEXT_FIELDS})
        return ans, {"model": "fake", "input_tokens": 1500, "output_tokens": 300, "ms": 7, "cost_usd": 0.012,
                     "cached": False}

    def add(self, truth: synth.CaseTruth) -> synth.CaseTruth:
        self.truths[truth.case_id] = truth
        return truth


@pytest.fixture
def model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(llm, "call_vision", fake)
    return fake


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CAI_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("CAI_IMAGES", str(tmp_path / "images"))
    monkeypatch.setenv("CAI_APP_DB", str(tmp_path / "app.db"))
    monkeypatch.setattr(llm, "CASSETTE", tmp_path / "cassette")
    monkeypatch.setattr(llm, "HOME", tmp_path)  # never pick up a real key from the repo's .env.local
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CAI_DEMO_PASSWORD", raising=False)
