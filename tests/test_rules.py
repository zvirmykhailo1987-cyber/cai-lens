from datetime import date

from cailens.agents import rules
from cailens.schema import Reading

D = date(2026, 6, 1)


def R(**kv):
    out = {}
    for k, v in kv.items():
        fid = k.replace("__", ".")
        out[fid] = Reading(field=fid, value=v, confidence=0.95, agent="test")
    return out


def base(**extra):
    kv = dict(A__firma="si", B__firma="si", feriti_no="si", feriti_si="no", data="20/05/2026",
              A__targa="AB123CD", B__targa="EF456GH", B__compagnia="POLARIS", A__cf="RSSMRA80A01H501U",
              A__totale="1", B__totale="1", A__c08="si", B__c01="si", A__cognome="ROSSI", A__compagnia="X",
              A__polizza="12345678", B__cognome="BIANCHI")
    kv.update(extra)
    return R(**kv)


def codes(r):
    return {i.code for i in rules.validate(r, D)}


def test_clean_form_has_no_issues_and_goes_through():
    r = base()
    assert codes(r) == set()
    dl = rules.deadline(r, D)
    assert dl.days == 30 and not dl.provisional
    route, reasons = rules.route(r, [], dl, True, True)
    assert route == "automatico", reasons


def test_declared_total_must_match_the_crosses():
    assert "totale_A_non_coincide" in codes(base(A__totale="2"))
    assert "totale_A_non_verificabile" in codes(base(A__totale="2", A__c03="incerto"))
    assert "totale_B_mancante" in codes(base(B__totale=None))


def test_signatures_and_deadline():
    assert rules.deadline(base(B__firma="no"), D).days == 60
    dl = rules.deadline(base(B__firma="incerto"), D)
    assert dl.days == 30 and dl.provisional  # uncertain: the shorter term, flagged
    assert rules.deadline(base(feriti_si="si", feriti_no="no"), D).days == 90
    assert "firma_mancante" in codes(base(A__firma="no"))
    assert "firma_incerta" in codes(base(A__firma="incerto"))


def test_injuries_box_problems():
    assert "feriti_contraddittorio" in codes(base(feriti_si="si", feriti_no="si"))
    r = base(feriti_si="no", feriti_no="no")
    assert "feriti_non_indicato" in codes(r)
    assert rules.deadline(r, D).provisional


def test_identifiers_dates_and_vehicle_b():
    assert "cf_non_valido" in codes(base(A__cf="RSSMRA80A01H501X"))
    assert "data_successiva" in codes(base(data="02/06/2026"))
    assert "data_non_valida" in codes(base(data="31/02/2026"))
    assert "dati_B_mancanti" in codes(base(B__targa=None))
    assert "targhe_uguali" in codes(base(B__targa="AB 123 CD"))
    assert "targa_formato" in codes(base(A__targa="ZH12345"))


def test_unilateral_report():
    assert "denuncia_unilaterale" in codes(base(B__firma="no", B__cognome=None))


def test_sketch_must_tell_the_same_story():
    r = base()
    r["scenario"] = Reading(field="scenario", value="rotatoria", confidence=0.9, agent="t")
    assert "grafico_incoerente" in codes(r)
    r["scenario"] = Reading(field="scenario", value="tamponamento", confidence=0.9, agent="t")
    assert "grafico_incoerente" not in codes(r)


def test_routing_reasons():
    r = base(A__c05="incerto")
    dl = rules.deadline(r, D)
    route, reasons = rules.route(r, rules.validate(r, D), dl, True, True)
    assert route == "revisione" and any("Caselle incerte" in x for x in reasons)
    assert rules.route({}, [], dl, False, False) == ("revisione", [rules.route({}, [], dl, False, False)[1][0]])
    route, reasons = rules.route(base(), [], rules.deadline(base(), D), True, False)
    assert route == "revisione" and any("non disponibile" in x for x in reasons)
    weak = base()
    weak["A.targa"] = Reading(field="A.targa", value="AB1Z3CD", confidence=0.4, agent="t")
    assert any("poco leggibili" in x for x in rules.route(weak, [], rules.deadline(weak, D), True, True)[1])
    assert any("qualità bassa" in x for x in rules.route(base(), [], rules.deadline(base(), D), True, True, "sfocata")[1])


def test_text_checks_are_skipped_without_handwriting():
    r = base(A__totale=None, B__targa=None, A__cf="BAD")
    assert codes(r) & {"totale_A_mancante", "dati_B_mancanti", "cf_non_valido"}
    only = {i.code for i in rules.validate(r, D, text_ok=False)}
    assert not only & {"totale_A_mancante", "dati_B_mancanti", "cf_non_valido"}
