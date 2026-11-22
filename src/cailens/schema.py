# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Contracts between the agents: the ground truth of a synthetic case, what each agent reads, and
the document-level result the router hands to a person."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Zone = Literal["anteriore", "posteriore", "lato_sx", "lato_dx", "ant_sx", "ant_dx", "post_sx", "post_dx"]
ZONES: tuple[Zone, ...] = ("anteriore", "posteriore", "lato_sx", "lato_dx", "ant_sx", "ant_dx", "post_sx", "post_dx")
Scenario = Literal["tamponamento", "incrocio", "cambio_corsia", "parcheggio", "retromarcia", "rotatoria"]
SCENARIOS: tuple[Scenario, ...] = ("tamponamento", "incrocio", "cambio_corsia", "parcheggio", "retromarcia", "rotatoria")
State = Literal["si", "no", "incerto"]  # a mark, a signature: present, absent, or "a person must look"


class Party(BaseModel):
    """What is written in column A or B of a synthetic form (the truth the evals compare against)."""

    cognome: str | None = None
    nome: str | None = None
    cf: str | None = None
    targa: str | None = None
    compagnia: str | None = None
    polizza: str | None = None
    cond_cognome: str | None = None
    cond_nome: str | None = None
    patente: str | None = None
    danni: str | None = None
    osservazioni: str | None = None
    impatto: Zone | None = None
    circostanze: list[int] = Field(default_factory=list)  # boxes really marked (1-17)
    annullate: list[int] = Field(default_factory=list)  # boxes marked then scribbled out: ambiguous by nature
    totale: str | None = None  # the declared number of marked boxes, as written
    firma: bool = True


class CaseTruth(BaseModel):
    case_id: str
    split: str
    seed: int
    data: str | None  # as written, dd/mm/yyyy
    ora: str | None
    luogo: str | None
    feriti: bool | None
    veicoli: bool | None  # damage to vehicles other than A and B
    oggetti: bool | None
    testimoni: str | None
    received_on: date  # when the insurer received the form (the deadline counts from here)
    scenario: Scenario
    A: Party
    B: Party
    defects: list[str] = Field(default_factory=list)
    capture: Literal["scansione", "foto"] = "scansione"


class Reading(BaseModel):
    """One field as read by one agent."""

    field: str
    value: str | None = None  # text value, or "si"/"no"/"incerto" for marks and signatures
    confidence: float = Field(ge=0, le=1)
    agent: str
    evidence: dict[str, float | str | None] = Field(default_factory=dict)  # e.g. ink ratio


class Issue(BaseModel):
    code: str
    severity: Literal["errore", "attenzione"]
    message: str
    fields: list[str] = Field(default_factory=list)


class Deadline(BaseModel):
    days: int | None
    due: date | None
    basis: str
    provisional: bool  # True when an uncertain input forced the shortest possible term
    human_only: bool = False  # art. 148 c.2-bis (suspected fraud) is never decided by code


class StepLog(BaseModel):
    step: str
    ms: int
    ok: bool
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    note: str = ""


class Result(BaseModel):
    case_id: str
    registered: bool
    registration_score: float
    readings: dict[str, Reading]
    issues: list[Issue]
    deadline: Deadline
    route: Literal["automatico", "revisione"]
    reasons: list[str]
    steps: list[StepLog]
    scenario: Scenario | None = None
    scenario_confidence: float = 0.0
    quality_note: str = ""  # why the photo is not trusted for marks, if it is not
    handwriting_ok: bool = False
    sharpness: float = 0.0
