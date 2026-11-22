# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""The one place that talks to the Claude API: images in, schema-valid JSON out, every call logged.

Responses are kept in a cassette (``evals/llm_cache``) keyed by purpose, prompt version, model and a
cache id: evals re-run for free and offline, CI never calls the API, and a reviewer can read exactly
what the model answered for each synthetic document.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cailens.paths import HOME

DEFAULT_MODEL = os.environ.get("CAI_MODEL", "claude-opus-5-5")
CASSETTE = Path(os.environ.get("CAI_CASSETTE", HOME / "evals" / "llm_cache"))
# USD per million tokens (input, output), first-party API list prices, Nov 2026
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}
_DROP = {"default", "maxLength", "minLength", "title", "examples"}

# Optional sink for every real call: (purpose, model, input_tokens, output_tokens, ms, ok)
RECORDER: Callable[[str, str, int, int, int, bool], None] | None = None


def load_local_env() -> None:
    """Read ANTHROPIC_API_KEY from this project's git-ignored .env.local, if the user put one there."""
    path = HOME / ".env.local"
    if path.exists() and "ANTHROPIC_API_KEY" not in os.environ:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()


class LLMUnavailable(RuntimeError):
    """No key, no network, a refusal or a malformed answer: the caller falls back to a person."""


def strict_schema(schema: Any) -> Any:
    """Objects get ``additionalProperties: false`` and every property required."""
    if isinstance(schema, list):
        return [strict_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: strict_schema(v) for k, v in schema.items() if k not in _DROP}
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        out["required"] = list(out["properties"])
    return out


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES.get(model, PRICES["claude-opus-5-5"])
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


def _key(purpose: str, prompt_version: str, model: str, cache_id: str) -> Path:
    digest = hashlib.sha256(f"{purpose}|{prompt_version}|{model}|{cache_id}".encode()).hexdigest()[:24]
    return CASSETTE / purpose / f"{digest}.json"


def call_vision(
    *,
    purpose: str,
    prompt_version: str,
    system: str,
    text: str,
    images: list[bytes],
    schema: dict[str, Any],
    cache_id: str | None = None,
    model: str | None = None,
    replay_only: bool = False,
    max_tokens: int = 4000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (answer, meta). ``meta`` has tokens, ms, cost and whether it came from the cassette."""
    model = model or DEFAULT_MODEL
    path = _key(purpose, prompt_version, model, cache_id) if cache_id else None
    if path and path.exists():
        hit = json.loads(path.read_text(encoding="utf-8"))
        return hit["answer"], {**hit["meta"], "cached": True}
    if replay_only or os.environ.get("CAI_REPLAY_ONLY") == "1":
        raise LLMUnavailable("risposta non in cache e chiamate al modello disattivate")
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        raise LLMUnavailable("nessuna chiave API configurata")
    import anthropic

    content: list[dict[str, Any]] = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": base64.standard_b64encode(img).decode()}}
        for img in images
    ]
    content.append({"type": "text", "text": text})
    t0 = time.perf_counter()
    usage = {"input_tokens": 0, "output_tokens": 0}
    ok = False
    try:
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"format": {"type": "json_schema", "schema": strict_schema(schema)}, "effort": "low"},
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",  # a refusal is retried on a fallback model chosen by the API
        }
        msg = anthropic.Anthropic(max_retries=3).beta.messages.create(**request)
        usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
        if msg.stop_reason == "refusal":
            raise LLMUnavailable("il modello ha rifiutato la richiesta")
        if msg.stop_reason == "max_tokens":
            raise LLMUnavailable(f"risposta troncata a {max_tokens} token")
        answer: dict[str, Any] = json.loads("".join(b.text for b in msg.content if b.type == "text"))
        ok = True
    except LLMUnavailable:
        raise
    except anthropic.APIError as exc:  # network, rate limit, server error after retries
        raise LLMUnavailable(f"{type(exc).__name__}: {str(exc)[:160]}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise LLMUnavailable(f"risposta non valida: {exc}") from exc
    finally:
        ms = int((time.perf_counter() - t0) * 1000)
        if RECORDER is not None:
            RECORDER(purpose, model, usage["input_tokens"], usage["output_tokens"], ms, ok)
    meta = {"model": msg.model, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
            "ms": ms, "cost_usd": round(cost_usd(model, **usage), 6), "cached": False,
            "image_sha": [hashlib.sha256(i).hexdigest()[:16] for i in images]}
    if path:
        # Keeping the answer is a convenience; losing a paid answer because the cassette is read-only (the demo
        # container runs with a read-only image) is not acceptable.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"purpose": purpose, "prompt_version": prompt_version, "cache_id": cache_id,
                                        "answer": answer, "meta": meta}, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            meta["cassette"] = "non scritta"
    return answer, meta
