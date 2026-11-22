import re

import pytest
from fastapi.testclient import TestClient

from cailens import synth, web
from cailens.appstate import AppState


@pytest.fixture
def app(model, monkeypatch, tmp_path):
    monkeypatch.setenv("CAI_DEMO_MODE", "1")
    monkeypatch.setenv("CAI_COOKIE_SECURE", "0")
    monkeypatch.setenv("CAI_REPLAY_ONLY", "0")
    monkeypatch.setattr(web, "DEMO_CASES", 4)
    # the fake model must know the demo cases' truth
    for s in synth.seeds("demo", 4):
        model.add(synth.make_case(s, "demo"))
    orig = synth.generate

    def gen(seed, split="dev"):
        t, img = orig(seed, split)
        model.add(t)
        return t, img

    monkeypatch.setattr(synth, "generate", gen)
    orig_custom = synth.custom

    def custom(seed, **kw):
        t, img = orig_custom(seed, **kw)
        return t, img

    monkeypatch.setattr(synth, "custom", custom)
    state = AppState(tmp_path / "web.db")
    monkeypatch.setattr(web, "seed_demo", lambda st, n=4: _seed(st, n))

    def _seed(st, n):
        for s in synth.seeds("demo", n):
            t, img = gen(s, "demo")
            web.store_case(st, t, img, scope="", source="demo", replay_only=False, use_llm=True)

    return TestClient(web.create_app(state)), state


def login(client):
    r = client.post("/accesso", data={"username": "liquidatore", "password": web.DEMO_PASSWORD, "next": "/coda"},
                    follow_redirects=False)
    assert r.status_code == 303
    page = client.get("/coda").text
    return re.search(r'name="csrf" value="([^"]+)"', page).group(1)


def test_pages_render_with_security_headers(app):
    client, _ = app
    for url in ("/", "/coda", "/qualita", "/processo", "/stato", "/genera", "/accesso", "/health"):
        r = client.get(url)
        assert r.status_code == 200, url
    r = client.get("/")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY"
    assert "style=" not in client.get("/coda").text  # the CSP forbids inline styles


def test_case_page_image_and_api(app):
    client, state = app
    cid = state.cases("")[0]["case_id"]
    page = client.get(f"/pratica/{cid}")
    assert (page.status_code == 200 and "Termine" in page.text) or "art. 148" in page.text
    assert client.get(f"/pratica/{cid}/pagina.jpg").headers["content-type"] == "image/jpeg"
    assert client.get(f"/pratica/{cid}/nulla.jpg").status_code == 404
    data = client.get(f"/api/pratica/{cid}").json()
    assert data["route"] in ("automatico", "revisione") and "A.firma" in data["fields"]
    assert client.get("/pratica/inesistente").status_code == 404


def test_review_needs_login_and_csrf(app):
    client, state = app
    cid = state.cases("")[0]["case_id"]
    r = client.post(f"/pratica/{cid}/campo", data={"field": "B.firma", "action": "correggi", "value": "si"},
                    follow_redirects=False)
    assert r.status_code == 303 and "/accesso" in r.headers["location"]
    login(client)
    r = client.post(f"/pratica/{cid}/campo", data={"field": "B.firma", "action": "correggi", "value": "si",
                                                  "csrf": "falso"})
    assert r.status_code == 403


def test_review_flow_and_visitor_isolation(app):
    client, state = app
    csrf = login(client)
    cid = state.cases("")[0]["case_id"]
    r = client.post(f"/pratica/{cid}/campo", data={"field": "B.firma", "action": "correggi", "value": "no",
                                                  "csrf": csrf}, follow_redirects=False)
    assert r.status_code == 303
    data = client.get(f"/api/pratica/{cid}").json()
    assert data["fields"]["B.firma"]["agent"] == "revisore" and data["deadline"]["days"] in (60, 90)
    assert client.post(f"/pratica/{cid}/campo", data={"field": "A.c01", "action": "correggi", "value": "forse",
                                                     "csrf": csrf}).status_code == 422
    assert client.post(f"/pratica/{cid}/campo", data={"field": "nulla", "action": "correggi", "value": "x",
                                                     "csrf": csrf}).status_code == 422
    assert client.post(f"/pratica/{cid}/campo", data={"field": "A.targa", "action": "boh", "csrf": csrf}).status_code == 422
    # another visitor of the shared account does not see the correction
    other = TestClient(client.app)
    login(other)
    assert other.get(f"/api/pratica/{cid}").json()["fields"]["B.firma"]["agent"] != "revisore"


def test_generate_a_case(app):
    client, _state = app
    assert client.post("/genera", data={"scenario": "incrocio"}, follow_redirects=False).status_code == 303
    csrf = login(client)
    assert client.post("/genera", data={"csrf": "x"}).status_code == 403
    r = client.post("/genera", data={"csrf": csrf, "scenario": "incrocio", "quality": "scansione",
                                     "defects": ["firma_B_mancante", "inventato"]}, follow_redirects=False)
    assert r.status_code == 303
    cid = r.headers["location"].rsplit("/", 1)[1]
    case = client.get(f"/api/pratica/{cid}").json()
    assert case["deadline"]["days"] == 60
    assert TestClient(client.app).get(f"/api/pratica/{cid}").status_code == 404  # anonymous visitors cannot see it
    assert client.post("/genera", data={"csrf": csrf, "scenario": "volo"}).status_code == 422


def test_generation_is_rate_limited(app, monkeypatch):
    client, _ = app
    csrf = login(client)
    codes = [client.post("/genera", data={"csrf": csrf, "quality": "scansione"}, follow_redirects=False).status_code
             for _ in range(6)]
    assert codes[:4] == [303] * 4 and codes[4] == 200  # the fifth gets the limit message
    assert "Troppe richieste" in client.post("/genera", data={"csrf": csrf}).text


def test_login_limiter_counts_only_failures(app):
    client, _ = app
    for _ in range(12):
        login(client)
    for _ in range(10):
        client.post("/accesso", data={"username": "liquidatore", "password": "sbagliata"})
    assert "Troppi tentativi" in client.post("/accesso", data={"username": "liquidatore", "password": "x"}).text


def test_logout_and_safe_next(app):
    client, _ = app
    csrf = login(client)
    assert client.post("/esci", data={"csrf": "no"}).status_code == 403
    assert client.post("/esci", data={"csrf": csrf}, follow_redirects=False).status_code == 303
    assert web.safe_next("//evil.example") == "/"
    assert web.safe_next("https://evil.example") == "/"
    assert web.safe_next("/coda?filtro=x") == "/coda?filtro=x"
    assert web.safe_next("/a\\b") == "/"


def test_operator_password_overrides_demo(model, monkeypatch, tmp_path):
    import secrets
    pw = secrets.token_urlsafe(12)
    monkeypatch.setenv("CAI_DEMO_PASSWORD", pw)
    client = TestClient(web.create_app(AppState(tmp_path / "op.db"), seed=False))
    assert client.post("/accesso", data={"username": "liquidatore", "password": web.DEMO_PASSWORD}).status_code == 200
    assert client.post("/accesso", data={"username": "liquidatore", "password": pw},
                       follow_redirects=False).status_code == 303
    assert web.DEMO_PASSWORD not in client.get("/accesso").text


def test_queue_filters(app):
    client, _ = app
    assert client.get("/coda?filtro=automatico").status_code == 200
    assert client.get("/coda?filtro=Firme da verificare").status_code == 200
