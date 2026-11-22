# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Web app: the liquidator's queue, one case with the form and what each agent read, a review that
recomputes rules and deadline, a page to build a synthetic case, evals and the process monitor."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections import Counter, OrderedDict, deque
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from cailens import form, llm, monitor, review, synth
from cailens.agents import rules
from cailens.agents.registrar import Registration
from cailens.appstate import AppState
from cailens.paths import DATA, HOME
from cailens.pipeline import process
from cailens.schema import SCENARIOS, CaseTruth, Result
from cailens.truth import expected

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))
SESSION_COOKIE = "cai_session"
DEMO_USER = "liquidatore"
DEMO_PASSWORD = "liquidatore-demo-2026"  # noqa: S105  # nosec B105 - public demo account, shown on sign-in
DEMO_CASES = int(os.environ.get("CAI_DEMO_CASES", "18"))
DEFECT_CHOICES = (
    ("firma_B_mancante", "Manca la firma di B"), ("totale_errato", "Totale delle caselle sbagliato"),
    ("correzione", "Una croce annullata a penna"), ("segno_debole", "Una croce piccola e leggera"),
    ("campo_B_mancante", "Targa o compagnia di B mancante"), ("cf_errato", "Codice fiscale scritto male"),
    ("denuncia_unilaterale", "B non ha compilato né firmato"), ("data_successiva", "Data dopo la ricezione"),
    ("firma_sigla", "B firma solo con una sigla"),
)
QUALITIES = (("scansione", "Scansione"), ("foto", "Foto con il telefono"), ("foto_scarsa", "Foto mossa e scura"))
FIELD_GROUPS = (
    ("Intestazione", ("data", "ora", "luogo", "testimoni", "feriti_no", "feriti_si", "veicoli_no", "veicoli_si",
                      "oggetti_no", "oggetti_si")),
    *((f"Veicolo {s}", tuple(f"{s}.{k}" for k in ("cognome", "nome", "cf", "targa", "compagnia", "polizza",
                                                     "cond_cognome", "cond_nome", "patente", "danni",
                                                     "osservazioni", "impatto", "totale", "firma")))
      for s in form.SIDES),
)
STATE_LABEL = {"si": "barrata", "no": "vuota", "incerto": "incerta"}
STATUS_LABEL = {"automatica": "Pronta, senza revisione", "pronta": "Pronta dopo revisione",
                "da_rivedere": "Da rivedere", "in_revisione": "In revisione", "da_leggere": "Non letta"}
STEP_LABEL = {"registrar": "Allineamento", "checkbox": "Caselle", "signature": "Firme", "handwriting": "Scrittura",
              "sketch": "Grafico", "verifier": "Seconda lettura", "regole": "Regole e termine",
              "baseline": "Solo modello"}


def field_label(fid: str) -> str:
    f = form.fields().get(fid)
    if f is not None:
        return f.label
    return {"scenario": "Dinamica (grafico)", "A.impatto": "Punto d'urto A", "B.impatto": "Punto d'urto B"}.get(fid, fid)


def show_value(fid: str, value: str | None) -> str:
    if value is None:
        return "—"
    f = form.fields().get(fid)
    if f is not None and f.kind in ("checkbox", "signature"):
        return {"si": "firmata", "no": "assente", "incerto": "incerta"}[value] if f.kind == "signature" \
            else STATE_LABEL.get(value, value)
    return value.replace("_", " ")


templates.env.filters["label"] = field_label
templates.env.globals["show_value"] = show_value
templates.env.globals["STEP_LABEL"] = STEP_LABEL
templates.env.globals["STATUS_LABEL"] = STATUS_LABEL


def it_date(d: Any) -> str:
    if isinstance(d, str):
        d = date.fromisoformat(d[:10])
    return d.strftime("%d/%m/%Y") if d else "—"


templates.env.filters["itdate"] = it_date


def pct0(v: float | None) -> str:
    if v is None:
        return "n/d"
    return f"{v:.1f}".rstrip("0").rstrip(".").replace(".", ",") + "%"


templates.env.filters["pct0"] = pct0


def safe_next(target: str) -> str:
    if not target.startswith("/") or any(ord(c) < 32 or c == "\\" for c in target):
        return "/"
    parts = urlsplit(target)
    return target if not parts.scheme and not parts.netloc and not target.startswith("//") else "/"


class RateLimiter:
    """Per-IP sliding window + per-IP and global daily caps, bounded memory (an LRU of addresses)."""

    def __init__(self, per_ip: int, window_s: int, daily: int, per_ip_daily: int = 25, max_ips: int = 10_000) -> None:
        self.per_ip, self.window_s, self.daily, self.per_ip_daily, self.max_ips = per_ip, window_s, daily, per_ip_daily, max_ips
        self.hits: OrderedDict[str, deque[float]] = OrderedDict()
        self.day_hits: Counter[str] = Counter()
        self.day, self.count = date.today(), 0
        self.lock = threading.Lock()

    def blocked(self, ip: str, now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        with self.lock:
            if self.count >= self.daily or self.day_hits[ip] >= self.per_ip_daily:
                return "Limite giornaliero raggiunto. Riprova domani."
            q = [t for t in self.hits.get(ip, ()) if now - t <= self.window_s]
            return "Troppi tentativi in poco tempo. Riprova tra qualche minuto." if len(q) >= self.per_ip else None

    def allow(self, ip: str, now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        with self.lock:
            if date.today() != self.day:
                self.day, self.count = date.today(), 0
                self.day_hits.clear()
            if self.count >= self.daily:
                return "Limite giornaliero della demo raggiunto. Riprova domani."
            if self.day_hits[ip] >= self.per_ip_daily:
                return "Hai raggiunto il limite giornaliero. Riprova domani."
            q = self.hits.pop(ip, deque())
            while q and now - q[0] > self.window_s:
                q.popleft()
            self.hits[ip] = q
            while len(self.hits) > self.max_ips:
                self.hits.popitem(last=False)
            if len(q) >= self.per_ip:
                return "Troppe richieste in poco tempo. Riprova tra qualche minuto."
            q.append(now)
            self.day_hits[ip] += 1
            self.count += 1
            return None


def images_dir() -> Path:
    d = Path(os.environ.get("CAI_IMAGES", DATA / "images"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def store_case(state: AppState, truth: CaseTruth, img: Image.Image, *, scope: str, source: str,
               replay_only: bool, use_llm: bool) -> Result:
    keep: list[Registration] = []
    result = process(img, truth.case_id, truth.received_on, cache_id=truth.case_id, replay_only=replay_only,
                     use_llm=use_llm, keep=keep)
    d = images_dir()
    img.save(d / f"{truth.case_id}.jpg", "JPEG", quality=85)
    if keep and keep[0].ok:
        Image.fromarray(keep[0].gray).save(d / f"{truth.case_id}-pagina.jpg", "JPEG", quality=85)
    state.save_case(truth.case_id, scope=scope, source=source, received_on=truth.received_on.isoformat(),
                    result=result.model_dump(mode="json"), truth=truth.model_dump(mode="json"))
    return result


def seed_demo(state: AppState, n: int = DEMO_CASES) -> None:
    """The shared demo queue: the same synthetic cases for every visitor (answers from the cassette)."""
    if state.count_cases("demo") >= n:
        return
    replay = os.environ.get("CAI_REPLAY_ONLY") == "1" or not os.environ.get("ANTHROPIC_API_KEY")
    for s in synth.seeds("demo", n):
        truth, img = synth.generate(s, "demo")
        store_case(state, truth, img, scope="", source="demo", replay_only=replay, use_llm=True)


def latest_results() -> dict[str, Any]:
    out: dict[str, Any] = {}
    base = HOME / "evals" / "results"
    for name in ("dev-llm-200", "holdout-llm-100", "dev-cv-200", "holdout-cv-100", "baseline-dev-40", "stress"):
        f = base / f"{name}.json"
        if f.exists():
            out[name] = json.loads(f.read_text(encoding="utf-8"))
    return out


def create_app(state: AppState | None = None, seed: bool = True) -> FastAPI:
    llm.load_local_env()  # local runs: the key from .env.local; in the container it comes from the environment
    app = FastAPI(title="CAI-Lens", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    st = state or AppState()
    st.ensure_user(DEMO_USER, os.environ.get("CAI_DEMO_PASSWORD", DEMO_PASSWORD),
                   reset="CAI_DEMO_PASSWORD" in os.environ)
    llm.RECORDER = st.log_llm
    demo_mode = os.environ.get("CAI_DEMO_MODE", "0") == "1"
    hint = f"{DEMO_USER} / {DEMO_PASSWORD}" if demo_mode and "CAI_DEMO_PASSWORD" not in os.environ else None
    secure_cookie = os.environ.get("CAI_COOKIE_SECURE", "1") == "1"
    login_limiter = RateLimiter(per_ip=10, window_s=900, daily=1_000_000, per_ip_daily=50)
    gen_limiter = RateLimiter(int(os.environ.get("CAI_GEN_PER_IP", "4")), 3600,
                              int(os.environ.get("CAI_GEN_DAILY", "40")), per_ip_daily=8)
    live_llm = bool(os.environ.get("ANTHROPIC_API_KEY")) and os.environ.get("CAI_REPLAY_ONLY") != "1"
    if seed:
        seed_demo(st)
    results = latest_results()

    @app.middleware("http")
    async def headers(request: Request, call_next: Any) -> Any:
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "same-origin"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        return resp

    def user_of(request: Request) -> dict[str, Any] | None:
        return st.session(request.cookies.get(SESSION_COOKIE))

    def scope_of(request: Request) -> str:
        """Each signed-in visitor of the public demo has a private scope (cases they built, their reviews)."""
        token = request.cookies.get(SESSION_COOKIE) or ""
        if not token or user_of(request) is None:
            return "anonimo"
        return hashlib.sha256(token.encode()).hexdigest()[:16] if demo_mode else "operatore"

    def page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"user": user_of(request), "demo_mode": demo_mode,
                                                          "live_llm": live_llm, "path": request.url.path, **ctx})

    def csrf_ok(user: dict[str, Any] | None, token: str) -> bool:
        return user is not None and hmac.compare_digest(token.encode(), user["csrf"].encode())

    def load(case_id: str, request: Request) -> tuple[dict[str, Any], Result, dict[str, Any]]:
        scope = scope_of(request)
        row = st.case(case_id, scope)
        if row is None:
            raise HTTPException(404, "Pratica non trovata")
        res = Result.model_validate(row["result"])
        eff = review.effective(res, st.reviews(case_id, scope), date.fromisoformat(row["received_on"]))
        return row, res, eff

    def queue(request: Request) -> list[dict[str, Any]]:
        scope = scope_of(request)
        today = date.today()
        out = []
        for row in st.cases(scope):
            res = Result.model_validate(row["result"])
            eff = review.effective(res, st.reviews(row["case_id"], scope), date.fromisoformat(row["received_on"]))
            due = eff["deadline"].due
            out.append({
                "id": row["case_id"], "source": row["source"], "received": row["received_on"],
                "route": eff["route"], "status": eff["status"], "reasons": eff["reasons"],
                "kinds": sorted({monitor.kind_of(r)[0] for r in eff["reasons"]}),
                "days": eff["deadline"].days, "due": due, "left": (due - today).days if due else None,
                "provisional": eff["deadline"].provisional, "to_look": len(review.needs_look(res, eff)),
                "plates": " · ".join(v for v in (res.readings.get("A.targa") and res.readings["A.targa"].value,
                                                  res.readings.get("B.targa") and res.readings["B.targa"].value) if v),
            })
        out.sort(key=lambda c: (c["route"] == "automatico", c["left"] if c["left"] is not None else 9999))
        return out

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        q = queue(request)
        return page(request, "home.html", results=results, cases=len(q),
                    auto=sum(c["route"] == "automatico" for c in q),
                    urgent=[c for c in q if c["route"] != "automatico"][:3])

    @app.get("/coda", response_class=HTMLResponse)
    def coda(request: Request, filtro: str = "") -> HTMLResponse:
        q = queue(request)
        kinds = Counter(k for c in q for k in c["kinds"])
        shown = [c for c in q if not filtro or filtro in c["kinds"] or (filtro == "automatico" and c["route"] == "automatico")]
        return page(request, "queue.html", queue=shown, total=len(q), kinds=kinds.most_common(), filtro=filtro,
                    state="ready" if shown else "empty",
                    auto=sum(c["route"] == "automatico" for c in q))

    @app.get("/pratica/{case_id}", response_class=HTMLResponse)
    def pratica(request: Request, case_id: str) -> HTMLResponse:
        row, res, eff = load(case_id, request)
        look = review.needs_look(res, eff)
        boxes = []
        for fid, r in eff["readings"].items():
            f = form.fields().get(fid)
            if f is None or f.kind == "sketch":
                continue
            tone = "ok"
            if fid in look or r.value == "incerto":
                tone = "look"
            if any(fid in i.fields and i.severity == "errore" for i in eff["issues"]):
                tone = "bad"
            if r.agent == review.REVIEWER:
                tone = "done"
            if f.kind in ("checkbox", "signature") and r.value == "no" and tone == "ok":
                continue  # empty boxes stay unmarked: the overlay shows what was read, not the whole grid
            if f.kind == "text" and r.value is None and tone == "ok":
                continue
            b = f.box
            boxes.append({"id": fid, "x": b.x0 - 3, "y": b.y0 - 3, "w": b.w + 6, "h": b.h + 6, "tone": tone})
        from cailens.evals import same  # local import: evals pulls in the whole pipeline

        truth_diff = []
        if row["truth"]:
            exp = expected(CaseTruth.model_validate(row["truth"]))
            for fid, want in exp.items():
                r = eff["readings"].get(fid)
                if r is None or r.confidence < rules.TEXT_MIN:
                    continue  # not a confident reading: it is already flagged for a person
                got = r.value
                if want == "annullata":
                    want = "incerto"
                if not same(fid, want, got) and got != "incerto":
                    truth_diff.append({"field": fid, "want": want, "got": got})
        has_page = (images_dir() / f"{case_id}-pagina.jpg").exists()
        return page(request, "case.html", row=row, res=res, eff=eff, look=look, boxes=boxes, groups=FIELD_GROUPS,
                    truth_diff=truth_diff, W=form.W, H=form.H, has_page=has_page,
                    csrf=(user_of(request) or {}).get("csrf"), cost=sum(s.cost_usd for s in res.steps))

    @app.get("/pratica/{case_id}/{kind}.jpg")
    def case_image(request: Request, case_id: str, kind: str) -> Response:
        if kind not in ("pagina", "originale"):
            raise HTTPException(404)
        load(case_id, request)  # scope check
        name = f"{case_id}-pagina.jpg" if kind == "pagina" else f"{case_id}.jpg"
        path = images_dir() / name
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/pratica/{case_id}/campo")
    def review_field(request: Request, case_id: str, field: str = Form(..., max_length=40),
                     action: str = Form(...), value: str = Form("", max_length=120), csrf: str = Form("")) -> Response:
        user = user_of(request)
        if user is None:
            return RedirectResponse(f"/accesso?next=/pratica/{case_id}", status_code=303)
        if not csrf_ok(user, csrf):
            raise HTTPException(403, "Token CSRF non valido")
        _row, _res, eff = load(case_id, request)
        f = form.fields().get(field)
        if field not in eff["readings"] and f is None:
            raise HTTPException(422, "Campo sconosciuto")
        old = eff["readings"][field].value if field in eff["readings"] else None
        if action == "conferma":
            new = old
        elif action == "correggi":
            new = value.strip().upper() or None
            if f is not None and f.kind in ("checkbox", "signature") and new not in ("SI", "NO"):
                raise HTTPException(422, "Valore non valido: si o no")
            if f is not None and f.kind in ("checkbox", "signature"):
                new = (new or "").lower()
        else:
            raise HTTPException(422, "Azione non valida")
        if action == "conferma" and old == "incerto":
            raise HTTPException(422, "Una lettura incerta va corretta, non confermata")
        st.record_review(case_id=case_id, scope=scope_of(request), field=field, old=old, new=new,
                         action="conferma" if action == "conferma" else "correzione", reviewer=user["username"])
        return RedirectResponse(f"/pratica/{case_id}#campi", status_code=303)

    @app.get("/genera", response_class=HTMLResponse)
    def genera_form(request: Request) -> HTMLResponse:
        return page(request, "generate.html", scenarios=SCENARIOS, defects=DEFECT_CHOICES, qualities=QUALITIES,
                    error=None, csrf=(user_of(request) or {}).get("csrf"))

    @app.post("/genera")
    async def genera_post(request: Request) -> Response:
        data = await request.form()
        user = user_of(request)
        if user is None:
            return RedirectResponse("/accesso?next=/genera", status_code=303)
        if not csrf_ok(user, str(data.get("csrf", ""))):
            raise HTTPException(403, "Token CSRF non valido")
        ip = request.client.host if request.client else "?"
        if (msg := gen_limiter.allow(ip)) is not None:
            return page(request, "generate.html", scenarios=SCENARIOS, defects=DEFECT_CHOICES, qualities=QUALITIES,
                        error=msg, csrf=user["csrf"])
        scenario = str(data.get("scenario", ""))
        quality = str(data.get("quality", "foto"))
        if (scenario and scenario not in SCENARIOS) or quality not in dict(QUALITIES):
            raise HTTPException(422, "Scelta non valida")
        allowed = {d for d, _ in DEFECT_CHOICES}
        chosen = [str(d) for d in data.getlist("defects") if str(d) in allowed]
        seed = 1_000_000 + secrets.randbelow(8_000_000)
        truth, img = synth.custom(seed, scenario=scenario or None, defects=chosen, quality=quality,
                                  feriti=data.get("feriti") == "si", received=date.today())
        scope = scope_of(request)
        truth.case_id = f"gen-{scope[:6]}-{seed}"
        store_case(st, truth, img, scope=scope, source="generata", replay_only=not live_llm, use_llm=True)
        return RedirectResponse(f"/pratica/{truth.case_id}", status_code=303)

    @app.get("/qualita", response_class=HTMLResponse)
    def qualita(request: Request) -> HTMLResponse:
        return page(request, "quality.html", r=results, state="ready" if results else "empty")

    @app.get("/processo", response_class=HTMLResponse)
    def processo(request: Request) -> HTMLResponse:
        scope = scope_of(request)
        cases = st.cases(scope)
        return page(request, "process.html", mon=monitor.summary(cases, st.events(scope), st.all_reviews(scope)))

    @app.get("/stato", response_class=HTMLResponse)
    def stato(request: Request) -> HTMLResponse:
        return page(request, "status.html", llm_stats=st.llm_stats(), model=llm.DEFAULT_MODEL,
                    prices=llm.PRICES.get(llm.DEFAULT_MODEL), gen_daily=gen_limiter.daily, gen_per_ip=gen_limiter.per_ip,
                    thresholds={"min_text": rules.TEXT_MIN})

    @app.get("/accesso", response_class=HTMLResponse)
    def login_form(request: Request, next: str = "/coda") -> HTMLResponse:
        return page(request, "login.html", next=safe_next(next), error=None, demo_password=hint)

    @app.post("/accesso", response_class=HTMLResponse)
    def login_post(request: Request, username: str = Form(..., max_length=40), password: str = Form(..., max_length=200),
                   next: str = Form("/coda")) -> Response:
        ip = request.client.host if request.client else "?"
        if (msg := login_limiter.blocked(ip)) is not None:
            return page(request, "login.html", next=safe_next(next), error=msg, demo_password=hint)
        got = st.login(username.strip(), password)
        if got is None:
            login_limiter.allow(ip)  # only failures count
            return page(request, "login.html", next=safe_next(next), error="Credenziali non valide.", demo_password=hint)
        resp = RedirectResponse(safe_next(next), status_code=303)
        resp.set_cookie(SESSION_COOKIE, got[0], httponly=True, secure=secure_cookie, samesite="lax", max_age=8 * 3600)
        return resp

    @app.post("/esci")
    def logout(request: Request, csrf: str = Form("")) -> Response:
        user = user_of(request)
        if user is not None and not csrf_ok(user, csrf):
            raise HTTPException(403, "Token CSRF non valido")
        st.logout(request.cookies.get(SESSION_COOKIE))
        resp = RedirectResponse("/", status_code=303)
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    @app.get("/api/pratica/{case_id}")
    def api_case(request: Request, case_id: str) -> JSONResponse:
        _, _res, eff = load(case_id, request)
        return JSONResponse({"case_id": case_id, "route": eff["route"], "status": eff["status"],
                             "reasons": eff["reasons"], "deadline": eff["deadline"].model_dump(mode="json"),
                             "issues": [i.model_dump() for i in eff["issues"]],
                             "fields": {k: r.model_dump(mode="json") for k, r in eff["readings"].items()}})

    return app


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8050")),  # noqa: S104  # nosec B104
                proxy_headers=True)
