from datetime import date

from conftest import case
from PIL import Image

from cailens import pipeline, review
from cailens.evals import score_case, summarize, true_deadline


def run(model, seed, split="dev", **kw):
    t, img = case(seed, split)
    model.add(t)
    return t, pipeline.process(img, t.case_id, t.received_on, cache_id=t.case_id, **kw)


def test_clean_case_goes_through_and_logs_every_step(model):
    # find a dev case without defects or injuries
    seed = next(s for s in range(10000, 10060) if not case(s)[0].defects and not case(s)[0].feriti)
    t, res = run(model, seed)
    assert res.route == "automatico", res.reasons
    assert [s.step for s in res.steps] == ["registrar", "checkbox", "signature", "handwriting", "sketch", "verifier", "regole"]
    assert res.deadline.days == true_deadline(t)
    assert sum(s.cost_usd for s in res.steps) > 0
    assert model.calls == ["handwriting", "sketch", "verifier"]


def test_second_reading_catches_a_plausible_digit_swap(model):
    seed = next(s for s in range(10000, 10060) if not case(s)[0].defects and not case(s)[0].feriti)
    t = case(seed)[0]
    swapped = t.A.targa.replace("3", "8") if "3" in t.A.targa else t.A.targa[:2] + "999" + t.A.targa[5:]
    model.mistakes["A.targa"] = swapped  # well-formed, passes every rule: only a second look sees it
    _, res = run(model, seed)
    assert res.route == "revisione"
    assert "Seconda lettura discordante" in res.reasons[0] and "A.targa" in res.reasons[0]
    assert res.readings["A.targa"].evidence["seconda_lettura"] == t.A.targa


def test_no_second_reading_no_automatic_route(model):
    seed = next(s for s in range(10000, 10060) if not case(s)[0].defects and not case(s)[0].feriti)
    model.verifier_down = True
    _, res = run(model, seed)
    assert res.route == "revisione"


def test_defects_are_flagged(model):
    t, res = run(model, 10003)  # totale_mancante + correzione
    assert res.route == "revisione"
    row = score_case(t, res)
    assert row["defect_hits"]["correzione"] is True
    assert row["wrong_critical"] == []


def test_model_mistake_is_caught_by_the_checksum(model):
    seed = next(s for s in range(10000, 10060) if not case(s)[0].defects)
    t = case(seed)[0]
    model.mistakes["A.cf"] = t.A.cf[:-1] + ("A" if t.A.cf[-1] != "A" else "B")
    _, res = run(model, seed)
    assert res.route == "revisione"
    assert any(i.code == "cf_non_valido" for i in res.issues)


def test_illegible_field_goes_to_review(model):
    seed = next(s for s in range(10000, 10060) if not case(s)[0].defects and not case(s)[0].feriti)
    model.illegible = {"B.targa"}
    _, res = run(model, seed)
    assert res.route == "revisione"
    assert res.readings["B.targa"].confidence < 0.5


def test_model_down_degrades_instead_of_failing(model):
    model.fail = True
    _t, res = run(model, 10000)
    assert res.route == "revisione"
    assert not res.handwriting_ok
    assert any("non disponibile" in r for r in res.reasons)
    assert res.deadline.days in (30, 60, 90)  # the deterministic part still works


def test_unreadable_image_stops_after_registration(model):
    res = pipeline.process(Image.new("RGB", (800, 1100), "white"), "vuoto", date(2026, 1, 1))
    assert not res.registered and res.route == "revisione" and res.readings == {}
    assert [s.step for s in res.steps] == ["registrar"]


def test_review_recomputes_deadline_and_route(model):
    seed = next(s for s in range(10000, 10060) if not case(s)[0].defects and not case(s)[0].feriti)
    t, res = run(model, seed)
    # pretend the signature agent was unsure: the term is provisional
    res.readings["B.firma"] = res.readings["B.firma"].model_copy(update={"value": "incerto", "confidence": 0.3})
    eff = review.effective(res, {}, t.received_on)
    assert eff["deadline"].provisional and eff["route"] == "revisione"
    assert "B.firma" in review.needs_look(res, eff)
    rv = {"B.firma": {"new_value": "si", "old_value": "incerto", "reviewer": "x", "at": "now"}}
    eff2 = review.effective(res, rv, t.received_on)
    assert eff2["deadline"].days == 30 and not eff2["deadline"].provisional
    assert eff2["status"] == "pronta"


def test_summary_counts(model):
    rows = []
    for s in (10000, 10001, 10003):
        t, res = run(model, s)
        rows.append(score_case(t, res))
    summ = summarize(rows)
    assert summ["documents"] == 3
    assert summ["checkbox"]["marked_read_no"] == 0 and summ["checkbox"]["empty_read_si"] == 0
    assert summ["text"]["exact_pct"] == 100.0
    assert summ["routing"]["automatico_with_wrong_key_field"] == 0


def test_review_view_keeps_the_second_readings_veto(model):
    """Found in the end-to-end run: a CF where the two readings differ is not a 'critical' text field, so
    recomputing the route from the readings alone used to show the case as automatic in queue, page and API."""
    from cailens.agents import rules, verifier

    fid = next(f for f in verifier.FIELDS if f not in rules.CRITICAL_TEXT and f.endswith(".cf"))
    seed = next(s for s in range(10000, 10060) if not case(s)[0].defects and not case(s)[0].feriti)
    t = case(seed)[0]
    true_cf = getattr(t, fid[0]).cf
    model.second_reading[fid] = true_cf[:-1] + ("A" if true_cf[-1] != "A" else "B")  # first reading is right
    _, res = run(model, seed)
    eff = review.effective(res, {}, t.received_on)
    assert eff["route"] == "revisione" and fid in eff["reasons"][0]
    assert fid in review.needs_look(res, eff)
    rv = {fid: {"new_value": true_cf, "old_value": res.readings[fid].value, "reviewer": "x", "at": "now"}}
    after = review.effective(res, rv, t.received_on)  # a person decided: the veto is lifted
    assert not any("Seconda lettura" in r for r in after["reasons"])
