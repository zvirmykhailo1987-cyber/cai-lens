from datetime import date

from hypothesis import given, settings
from hypothesis import strategies as st

from cailens import formats, mcp_server
from cailens.agents import rules
from cailens.schema import Reading

STATES = st.sampled_from(["si", "no", "incerto"])


@given(a=STATES, b=STATES, inj=STATES)
@settings(max_examples=200, deadline=None)
def test_deadline_is_never_later_than_any_term_the_facts_allow(a, b, inj):
    out = mcp_server.deadline_148(a, b, inj, "2026-03-01")
    # the possible true terms given the uncertainty
    sigs = [[x] if x != "incerto" else ["si", "no"] for x in (a, b)]
    injs = [inj] if inj != "incerto" else ["si", "no"]
    possible = {90 if i == "si" else (30 if sa == "si" and sb == "si" else 60) for sa in sigs[0] for sb in sigs[1] for i in injs}
    if inj == "si":
        assert out["days"] == 90
    else:
        assert out["days"] <= min(possible - {90}) if possible - {90} else True
    if inj != "si" and "incerto" in (a, b, inj):  # uncertainty that can change the term is always flagged
        assert out["provisional"]
    if inj == "si":
        assert not out["provisional"]  # with injuries the signatures do not matter: 90 days, final


@given(st.text(max_size=20))
def test_format_checks_never_crash(s):
    formats.shape_ok("A.targa", s)
    formats.shape_ok("data", s)
    formats.cf_valid(s)
    formats.parse_date(s)


@given(total=st.integers(0, 17), crosses=st.sets(st.integers(1, 17), max_size=17))
def test_total_check_matches_the_count(total, crosses):
    r = {f"A.c{n:02d}": Reading(field=f"A.c{n:02d}", value="si" if n in crosses else "no", confidence=1, agent="t")
         for n in range(1, 18)}
    r["A.totale"] = Reading(field="A.totale", value=str(total), confidence=1, agent="t")
    flagged = any(i.code == "totale_A_non_coincide" for i in rules.validate(r, date(2026, 1, 1)))
    assert flagged == (total != len(crosses))


def test_mcp_tools(tmp_path, monkeypatch, model):
    monkeypatch.setenv("CAI_MCP_ROOT", str(tmp_path / "inbox"))
    made = mcp_server.make_synthetic_case(42)
    from cailens.schema import CaseTruth
    t = CaseTruth.model_validate(made["truth"])
    model.truths[made["image_path"].removesuffix(".jpg")] = t
    out = mcp_server.read_cai(made["image_path"], made["received_on"])
    assert out["route"] in ("automatico", "revisione") and "A.firma" in out["fields"]
    assert "error" in mcp_server.read_cai("../../etc/passwd", "2026-01-01")
    outside = tmp_path / "fuori.jpg"
    (tmp_path / "inbox" / made["image_path"]).replace(outside)
    assert "deve stare" in mcp_server.read_cai(str(outside), made["received_on"])["error"]
    outside.replace(tmp_path / "inbox" / made["image_path"])
    assert "error" in mcp_server.read_cai(made["image_path"], "ieri")
    assert "error" in mcp_server.read_cai("manca.jpg", "2026-01-01")
    (tmp_path / "inbox" / "rotto.jpg").write_bytes(b"\xff\xd8\xff not a jpeg")
    assert "non leggibile" in mcp_server.read_cai("rotto.jpg", "2026-01-01")["error"]
    assert mcp_server.check_codice_fiscale("RSSMRA80A01H501U")["check_char_ok"]
    assert not mcp_server.check_codice_fiscale("RSSMRA80A01H501X")["check_char_ok"]
    assert "error" in mcp_server.deadline_148("si", "si", "no", "x")
