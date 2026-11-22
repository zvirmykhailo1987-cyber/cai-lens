# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Writable state of the web app (SQLite): users, sessions, cases, step events, reviews, LLM calls.

Reviews are an append-only log: a correction never overwrites what the agents read, it is added on top,
with who and when. ``scope`` keeps visitors of the public demo apart (they share one sign-in).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cailens.paths import DATA

SESSION_HOURS = 8
SCHEMA = """
create table if not exists users (
    username text primary key, pw_hash text not null, role text not null);
create table if not exists sessions (
    token_hash text primary key, username text not null references users(username), csrf text not null,
    expires_at text not null);
create table if not exists cases (
    case_id text primary key, scope text not null default '', source text not null, created_at text not null,
    received_on text not null, result_json text not null, truth_json text);
create index if not exists cases_scope on cases(scope);
create table if not exists events (
    id integer primary key autoincrement, case_id text not null, scope text not null default '', step text not null,
    ms integer not null, ok integer not null, input_tokens integer not null default 0,
    output_tokens integer not null default 0, cost_usd real not null default 0, at text not null);
create table if not exists reviews (
    id integer primary key autoincrement, case_id text not null, scope text not null default '', field text not null,
    old_value text, new_value text, action text not null check (action in ('conferma', 'correzione')),
    reviewer text not null, at text not null);
create index if not exists reviews_case on reviews(case_id, scope);
create table if not exists llm_calls (
    id integer primary key autoincrement, at text not null, purpose text not null, model text not null,
    input_tokens integer not null, output_tokens integer not null, ms integer not null, ok integer not null);
"""


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    candidate = hash_password(password, bytes.fromhex(salt_hex)).split("$")[2]
    return hmac.compare_digest(candidate, digest_hex)


_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp() -> str:
    return _now().isoformat(timespec="seconds")


class AppState:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(os.environ.get("CAI_APP_DB", DATA / "app.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self.db() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def db(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            con = sqlite3.connect(self.path)
            con.row_factory = sqlite3.Row
            try:
                yield con
                con.commit()
            finally:
                con.close()

    # --- users and sessions ------------------------------------------------------------------
    def ensure_user(self, username: str, password: str, role: str = "liquidatore", reset: bool = False) -> None:
        with self.db() as con:
            row = con.execute("select pw_hash from users where username=?", [username]).fetchone()
            if row is None:
                con.execute("insert into users values (?,?,?)", [username, hash_password(password), role])
            elif reset and not check_password(password, row["pw_hash"]):
                con.execute("update users set pw_hash=? where username=?", [hash_password(password), username])
                con.execute("delete from sessions where username=?", [username])

    def login(self, username: str, password: str) -> tuple[str, str] | None:
        with self.db() as con:
            row = con.execute("select pw_hash from users where username=?", [username]).fetchone()
            stored = row["pw_hash"] if row is not None else _DUMMY_HASH  # same cost either way
            if not check_password(password, stored) or row is None:
                return None
            token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
            con.execute("delete from sessions where expires_at < ?", [_stamp()])
            con.execute("insert into sessions values (?,?,?,?)",
                        [hashlib.sha256(token.encode()).hexdigest(), username, csrf,
                         (_now() + timedelta(hours=SESSION_HOURS)).isoformat(timespec="seconds")])
            return token, csrf

    def session(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        with self.db() as con:
            row = con.execute(
                "select s.username, s.csrf, s.expires_at, u.role from sessions s join users u using (username) "
                "where s.token_hash=?", [hashlib.sha256(token.encode()).hexdigest()]).fetchone()
        if row is None or row["expires_at"] < _stamp():
            return None
        return dict(row)

    def logout(self, token: str | None) -> None:
        if token:
            with self.db() as con:
                con.execute("delete from sessions where token_hash=?", [hashlib.sha256(token.encode()).hexdigest()])

    # --- cases and events ---------------------------------------------------------------------
    def save_case(self, case_id: str, *, scope: str, source: str, received_on: str, result: dict[str, Any],
                  truth: dict[str, Any] | None) -> None:
        with self.db() as con:
            con.execute("insert or replace into cases values (?,?,?,?,?,?,?)",
                        [case_id, scope, source, _stamp(), received_on, json.dumps(result, default=str),
                         json.dumps(truth, default=str) if truth else None])
            con.execute("delete from events where case_id=?", [case_id])
            con.executemany(
                "insert into events (case_id, scope, step, ms, ok, input_tokens, output_tokens, cost_usd, at) "
                "values (?,?,?,?,?,?,?,?,?)",
                [(case_id, scope, s["step"], s["ms"], int(s["ok"]), s.get("input_tokens", 0),
                  s.get("output_tokens", 0), s.get("cost_usd", 0.0), _stamp()) for s in result.get("steps", [])])

    def cases(self, scope: str) -> list[dict[str, Any]]:
        """The shared demo queue (scope '') plus this visitor's own cases."""
        with self.db() as con:
            rows = con.execute("select * from cases where scope in ('', ?) order by received_on, case_id",
                               [scope]).fetchall()
        return [self._row(r) for r in rows]

    def case(self, case_id: str, scope: str) -> dict[str, Any] | None:
        with self.db() as con:
            row = con.execute("select * from cases where case_id=? and scope in ('', ?)", [case_id, scope]).fetchone()
        return self._row(row) if row else None

    def count_cases(self, source: str) -> int:
        with self.db() as con:
            return int(con.execute("select count(*) from cases where source=?", [source]).fetchone()[0])

    @staticmethod
    def _row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        d["result"] = json.loads(d.pop("result_json"))
        d["truth"] = json.loads(d.pop("truth_json")) if d.get("truth_json") else None
        return d

    def events(self, scope: str) -> list[dict[str, Any]]:
        with self.db() as con:
            return [dict(r) for r in con.execute("select * from events where scope in ('', ?)", [scope]).fetchall()]

    # --- reviews (append-only) -----------------------------------------------------------------
    def record_review(self, *, case_id: str, scope: str, field: str, old: str | None, new: str | None,
                      action: str, reviewer: str) -> None:
        if action not in ("conferma", "correzione"):
            raise ValueError("action must be 'conferma' or 'correzione'")
        with self.db() as con:
            con.execute("insert into reviews (case_id, scope, field, old_value, new_value, action, reviewer, at) "
                        "values (?,?,?,?,?,?,?,?)", [case_id, scope, field, old, new, action, reviewer, _stamp()])

    def reviews(self, case_id: str, scope: str) -> dict[str, dict[str, Any]]:
        with self.db() as con:
            rows = con.execute("select * from reviews where case_id=? and scope=? order by id", [case_id, scope]).fetchall()
        return {r["field"]: dict(r) for r in rows}  # the last word on each field wins

    def all_reviews(self, scope: str) -> list[dict[str, Any]]:
        with self.db() as con:
            return [dict(r) for r in con.execute("select * from reviews where scope=? order by id", [scope]).fetchall()]

    # --- LLM call log ---------------------------------------------------------------------------
    def log_llm(self, purpose: str, model: str, input_tokens: int, output_tokens: int, ms: int, ok: bool) -> None:
        with self.db() as con:
            con.execute("insert into llm_calls (at, purpose, model, input_tokens, output_tokens, ms, ok) "
                        "values (?,?,?,?,?,?,?)", [_stamp(), purpose, model, input_tokens, output_tokens, ms, int(ok)])

    def llm_stats(self, days: int = 30) -> list[dict[str, Any]]:
        since = (_now() - timedelta(days=days)).isoformat()
        with self.db() as con:
            rows = con.execute(
                "select purpose, model, count(*) n, sum(input_tokens) i, sum(output_tokens) o, avg(ms) ms, "
                "sum(1 - ok) errors from llm_calls where at >= ? group by purpose, model order by n desc", [since]).fetchall()
        return [dict(r) for r in rows]
