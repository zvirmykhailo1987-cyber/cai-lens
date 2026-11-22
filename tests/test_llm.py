import json
import sys
import types

import pytest

from cailens import llm


def test_strict_schema_closes_objects():
    s = llm.strict_schema({"type": "object", "title": "x", "properties": {"a": {"type": "string", "default": "z"},
                                                                         "b": {"type": "object", "properties": {"c": {"type": "integer"}}}}})
    assert s["additionalProperties"] is False and s["required"] == ["a", "b"]
    assert "default" not in s["properties"]["a"] and s["properties"]["b"]["required"] == ["c"]


def test_cost():
    assert llm.cost_usd("claude-opus-5-5", 1_000_000, 0) == 4.0
    assert llm.cost_usd("claude-sonnet-5-5", 0, 1_000_000) == 10.0


def test_without_key_or_in_replay_mode_it_refuses():
    kw = dict(purpose="p", prompt_version="v", system="s", text="t", images=[b"x"], schema={"type": "object"})
    with pytest.raises(llm.LLMUnavailable, match="chiave"):
        llm.call_vision(**kw)
    with pytest.raises(llm.LLMUnavailable, match="cache"):
        llm.call_vision(**kw, cache_id="c1", replay_only=True)


def test_cassette_is_replayed(tmp_path):
    path = llm._key("p", "v", llm.DEFAULT_MODEL, "c1")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"answer": {"ok": 1}, "meta": {"ms": 3, "cost_usd": 0.01}}), encoding="utf-8")
    ans, meta = llm.call_vision(purpose="p", prompt_version="v", system="s", text="t", images=[b"x"],
                                schema={"type": "object"}, cache_id="c1", replay_only=True)
    assert ans == {"ok": 1} and meta["cached"] is True


class _Msg:
    def __init__(self, text, stop="end_turn"):
        self.stop_reason = stop
        self.model = "claude-opus-5-5"
        self.usage = types.SimpleNamespace(input_tokens=1000, output_tokens=200)
        self.content = [types.SimpleNamespace(type="text", text=text)]


def _fake_anthropic(monkeypatch, msg, seen):
    class Messages:
        def create(self, **kw):
            seen.append(kw)
            if isinstance(msg, Exception):
                raise msg
            return msg

    class Client:
        def __init__(self, **kw):
            self.beta = types.SimpleNamespace(messages=Messages())

    mod = types.ModuleType("anthropic")
    mod.Anthropic = Client
    mod.APIError = type("APIError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    return mod


def test_live_call_is_recorded_and_cached(monkeypatch):
    seen, rec = [], []
    _fake_anthropic(monkeypatch, _Msg('{"a": "b"}'), seen)
    monkeypatch.setattr(llm, "RECORDER", lambda *a: rec.append(a))
    ans, meta = llm.call_vision(purpose="p", prompt_version="v", system="s", text="t", images=[b"img"],
                                schema={"type": "object", "properties": {"a": {"type": "string"}}}, cache_id="k")
    assert ans == {"a": "b"} and meta["input_tokens"] == 1000 and meta["cost_usd"] > 0
    req = seen[0]
    assert req["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert req["messages"][0]["content"][0]["type"] == "image" and req["fallbacks"] == "default"
    assert rec and rec[0][-1] is True
    again, meta2 = llm.call_vision(purpose="p", prompt_version="v", system="s", text="t", images=[b"img"],
                                   schema={"type": "object"}, cache_id="k")
    assert again == ans and meta2["cached"] and len(seen) == 1


@pytest.mark.parametrize("msg,match", [(_Msg("{}", "refusal"), "rifiutato"), (_Msg("{", "max_tokens"), "troncata"),
                                       (_Msg("not json"), "non valida")])
def test_bad_answers_become_unavailable(monkeypatch, msg, match):
    _fake_anthropic(monkeypatch, msg, [])
    with pytest.raises(llm.LLMUnavailable, match=match):
        llm.call_vision(purpose="p", prompt_version="v", system="s", text="t", images=[b"i"], schema={"type": "object"})


def test_api_error_becomes_unavailable(monkeypatch):
    mod = _fake_anthropic(monkeypatch, None, [])
    err = mod.APIError("boom")
    _fake_anthropic(monkeypatch, err, [])
    sys.modules["anthropic"].APIError = type(err)
    with pytest.raises(llm.LLMUnavailable):
        llm.call_vision(purpose="p", prompt_version="v", system="s", text="t", images=[b"i"], schema={"type": "object"})


def test_read_only_cassette_does_not_lose_a_paid_answer(monkeypatch, tmp_path):
    """The demo container's image is read-only: the answer must still come back, only the cassette is skipped."""
    seen = []
    _fake_anthropic(monkeypatch, _Msg('{"a": "b"}'), seen)
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(llm, "CASSETTE", blocker / "cassette")  # mkdir under a file -> OSError, as on a read-only fs
    ans, meta = llm.call_vision(purpose="p", prompt_version="v", system="s", text="t", images=[b"img"],
                                schema={"type": "object"}, cache_id="ro")
    assert ans == {"a": "b"} and meta["cassette"] == "non scritta" and len(seen) == 1
