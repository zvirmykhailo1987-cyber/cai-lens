"""End-to-end check of a running CAI-Lens demo, from every side: visitor, two reviewers, attacker, live cases.

    python scripts/e2e_check.py http://localhost:8050            # pages, isolation, security
    python scripts/e2e_check.py URL --live 2                     # + 2 cases read by the real model (costs ~$0.04 each)
    python scripts/e2e_check.py URL --limits                     # + rate limits (blocks this IP for a while)

Run it against the production container before a release: it found two bugs the unit tests had not
(a live answer lost on a read-only image, and the review view dropping the verifier's veto).
"""

from __future__ import annotations

import re
import sys
import time

import httpx

BASE = sys.argv[1].rstrip("/")
LIVE = int(sys.argv[sys.argv.index("--live") + 1]) if "--live" in sys.argv else 0
LIMITS = "--limits" in sys.argv
DEMO = {"username": "liquidatore", "password": "liquidatore-demo-2026"}
OK: list[str] = []
BAD: list[str] = []


def check(name: str, cond: bool, info: str = "") -> bool:
    (OK if cond else BAD).append(f"{name} {info}".strip())
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{info}]" if info else ""))
    return cond


def client() -> httpx.Client:
    return httpx.Client(base_url=BASE, follow_redirects=False, timeout=180)


def csrf_of(html: str) -> str:
    m = re.search(r'name="csrf" value="([^"]+)"', html)
    return m.group(1) if m else ""


def login(c: httpx.Client) -> httpx.Response:
    return c.post("/accesso", data={**DEMO, "next": "/coda"})


print("== 1. anonymous visitor")
anon = client()
check("health", anon.get("/health").json() == {"status": "ok"})
for p in ["/", "/coda", "/genera", "/qualita", "/processo", "/stato", "/accesso"]:
    r = anon.get(p)
    check(f"GET {p}", r.status_code == 200, str(r.status_code))
h = anon.get("/").headers
check("CSP header", "default-src 'self'" in h.get("content-security-policy", ""))
check("X-Frame-Options DENY", h.get("x-frame-options") == "DENY")
check("nosniff", h.get("x-content-type-options") == "nosniff")
ids = sorted(set(re.findall(r'href="/pratica/([\w-]+)"', anon.get("/coda").text)))
check("demo queue has 18 cases", len(ids) == 18, str(len(ids)))
check("every demo case opens", all(anon.get(f"/pratica/{c}").status_code == 200 for c in ids))
cid = ids[0]
check("straightened page image", anon.get(f"/pratica/{cid}/pagina.jpg").headers.get("content-type") == "image/jpeg")
check("original image", anon.get(f"/pratica/{cid}/originale.jpg").status_code == 200)
check("API returns route + deadline", {"route", "deadline"} <= set(anon.get(f"/api/pratica/{cid}").json()))
css = re.findall(r'href="(/static/[^"]+\.css)', anon.get("/").text)
check("stylesheet", bool(css) and all(anon.get(u).status_code == 200 for u in css), str(css))
fonts = re.findall(r'url\("?([^")]+\.woff2)', anon.get(css[0]).text) if css else []
check("fonts", bool(fonts) and all(anon.get("/static/" + f).status_code == 200 for f in fonts), f"{len(fonts)} fonts")
check("unknown case 404", anon.get("/pratica/nope-123").status_code == 404)
check("other file names 404", anon.get(f"/pratica/{cid}/secret.jpg").status_code == 404)
check("path traversal 404", anon.get("/pratica/..%2f..%2fetc%2fpasswd/pagina.jpg").status_code == 404)
check("static traversal 404", anon.get("/static/..%2f..%2fweb.py").status_code == 404)
check("API docs off", anon.get("/docs").status_code == 404 and anon.get("/openapi.json").status_code == 404)
r = anon.post("/genera", data={"quality": "foto"})
check("anonymous generate -> sign in", r.status_code == 303 and "/accesso" in r.headers["location"])
r = anon.post(f"/pratica/{cid}/campo", data={"field": "A.targa", "action": "conferma"})
check("anonymous review -> sign in", r.status_code == 303 and "/accesso" in r.headers["location"])
check("sign-in page shows the demo credentials", DEMO["password"] in anon.get("/accesso").text)

print("== 2. two reviewers on the shared demo account")
v1, v2 = client(), client()
r = login(v1)
check("visitor 1 signs in", r.status_code == 303 and r.headers["location"] == "/coda")
ck = r.headers.get("set-cookie", "").lower()
check("cookie HttpOnly + SameSite", "httponly" in ck and "samesite=lax" in ck)
if BASE.startswith("https://"):
    check("cookie Secure over HTTPS", "secure" in ck)
check("visitor 2 signs in", login(v2).status_code == 303)
review_ids = [i for i in ids if 'name="field"' in v1.get(f"/pratica/{i}").text]
check("cases offer field review", bool(review_ids), str(len(review_ids)))
cid = review_ids[0] if review_ids else cid
p1 = v1.get(f"/pratica/{cid}")
tok1 = csrf_of(p1.text)
check("visitor 1 still signed in after visitor 2", p1.status_code == 200 and tok1 != "")
field = (re.findall(r'name="field" value="([^"]+)"', p1.text) or ["A.targa"])[0]
r = v1.post(f"/pratica/{cid}/campo", data={"field": field, "action": "correggi", "value": "no", "csrf": tok1})
if r.status_code == 422:
    r = v1.post(f"/pratica/{cid}/campo", data={"field": field, "action": "correggi", "value": "AB123CD", "csrf": tok1})
check("visitor 1 review accepted", r.status_code == 303, f"{r.status_code} {field}")
a0, a1, a2 = (c.get(f"/api/pratica/{cid}").json() for c in (anon, v1, v2))
check("review visible to visitor 1", a1["fields"][field] != a0["fields"][field] or a1["status"] != a0["status"])
check("review NOT visible to visitor 2", a2["fields"][field] == a0["fields"][field])

print("== 3. attacker")
campo = f"/pratica/{cid}/campo"
check("no CSRF token -> 403", v1.post(campo, data={"field": field, "action": "conferma"}).status_code == 403)
check("wrong CSRF token -> 403", v1.post(campo, data={"field": field, "action": "conferma", "csrf": "x" * 40}).status_code == 403)
tok2 = csrf_of(v2.get(f"/pratica/{cid}").text)
check("another visitor's token -> 403", v1.post(campo, data={"field": field, "action": "conferma", "csrf": tok2}).status_code == 403)
check("generate without CSRF -> 403", v1.post("/genera", data={"quality": "foto"}).status_code == 403)
check("unknown field 422", v1.post(campo, data={"field": "zzz", "action": "conferma", "csrf": tok1}).status_code == 422)
check("overlong field 422", v1.post(campo, data={"field": "x" * 500, "action": "conferma", "csrf": tok1}).status_code == 422)
check("bad action 422", v1.post(campo, data={"field": field, "action": "drop", "csrf": tok1}).status_code == 422)
check("logout with wrong CSRF 403", v1.post("/esci", data={"csrf": "nope"}).status_code == 403)
r = client().post("/accesso", data={**DEMO, "next": "//evil.example"})
check("open redirect blocked", r.headers.get("location") == "/", r.headers.get("location", ""))
check("wrong password rejected", "Credenziali non valide" in client().post("/accesso", data={**DEMO, "password": "x"}).text)
check("forged cookie = anonymous", client().get("/coda", headers={"Cookie": "cai_session=forged"}).status_code == 200)
v1.post(campo, data={"field": "A.targa", "action": "correggi", "value": "<script>alert(1)</script>", "csrf": tok1})
check("XSS in a correction is escaped", "<script>alert(1)" not in v1.get(f"/pratica/{cid}").text)

if LIVE:
    print(f"== 4. live cases ({LIVE}, real model)")
    g = v1.get("/genera").text
    scen = re.findall(r'name="scenario" value="([^"]+)"', g)
    defects = re.findall(r'name="defects" value="([^"]+)"', g)
    for i in range(LIVE):
        data = {"csrf": csrf_of(g), "scenario": scen[i % len(scen)] if scen else "", "quality": "foto",
                "defects": defects[:1] if i == 0 else []}
        t0 = time.time()
        r = v1.post("/genera", data=data)
        if not check(f"live case #{i + 1}", r.status_code == 303, f"{r.status_code} in {time.time() - t0:.1f}s"):
            print(r.text[:600])
            continue
        loc = r.headers["location"]
        js = v1.get(f"/api/pratica/{loc.rsplit('/', 1)[-1]}").json()
        check("generated case opens", v1.get(loc).status_code == 200)
        check("handwriting read by the model", bool(js["fields"].get("A.targa", {}).get("value")))
        check("generated case private to its visitor", v2.get(loc).status_code == 404 and anon.get(loc).status_code == 404)
        if data["defects"]:
            check("chosen defect is flagged", js["route"] == "revisione" or bool(js["issues"]), str(data["defects"]))
        print("       route:", js["route"], "| reasons:", js["reasons"][:3])
    check("Stato shows the model", "claude-opus-5-5" in v1.get("/stato").text)

if LIMITS:
    print("== 5. limits")
    lim = client()
    login(lim)
    tok = csrf_of(lim.get("/genera").text)
    codes = []
    for i in range(6):
        r = lim.post("/genera", data={"csrf": tok, "quality": "foto"}, headers={"X-Forwarded-For": f"10.0.0.{i}"})
        codes.append("303" if r.status_code == 303 else ("limit" if "limite" in r.text.lower() or "Troppe" in r.text else str(r.status_code)))
    check("4 cases per IP per hour, X-Forwarded-For spoof ignored", codes[:4] == ["303"] * 4 and codes[4] == "limit", str(codes))
    bf = client()
    res = ["blocked" if "Troppi tentativi" in bf.post("/accesso", data={**DEMO, "password": f"w{i}"}).text else "denied"
           for i in range(12)]
    first = res.index("blocked") if "blocked" in res else 99
    check("password guessing blocked within 10 failures", first <= 10 and set(res[first:]) == {"blocked"}, str(res))
    check("blocked IP cannot sign in even with the right password", "Troppi" in bf.post("/accesso", data=DEMO).text)

print(f"\nRESULT: {len(OK)} ok, {len(BAD)} failed")
for b in BAD:
    print("  FAILED:", b)
sys.exit(1 if BAD else 0)
