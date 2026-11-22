from datetime import date

from cailens import formats, synth
from cailens.synth import codice_fiscale, make_case, plate


def test_case_is_deterministic():
    assert make_case(10003, "dev") == make_case(10003, "dev")
    assert make_case(10003, "dev") != make_case(10004, "dev")


def test_identifiers_have_valid_shapes():
    for s in range(10000, 10040):
        t = make_case(s, "dev")
        if "cf_errato" not in t.defects:
            assert formats.cf_valid(t.A.cf), t.A.cf
        if t.B.cf:
            assert formats.cf_valid(t.B.cf)
        for p in (t.A.targa, t.B.targa):
            if p:
                assert formats.shape_ok("A.targa", p)


def test_known_codice_fiscale():
    # the classic textbook example: Mario Rossi, born 1 Jan 1980 in Roma (H501)
    assert codice_fiscale("ROSSI", "MARIO", date(1980, 1, 1), False, "H501") == "RSSMRA80A01H501U"


def test_plate_never_uses_forbidden_letters():
    import random
    rng = random.Random(1)
    for _ in range(200):
        assert not set(plate(rng)) & set("IOQU")


def test_defects_change_the_truth():
    t = make_case(1, "dev", defects=["firma_B_mancante", "cf_errato", "campo_B_mancante"])
    assert t.B.firma is False
    assert not formats.cf_valid(t.A.cf)
    assert t.B.targa is None or t.B.compagnia is None
    u = make_case(2, "dev", defects=["denuncia_unilaterale"])
    assert u.B.cognome is None and u.B.targa and u.B.firma is False
    w = make_case(3, "dev", defects=["totale_errato"])
    side = w.A if w.A.totale != str(len(w.A.circostanze)) else w.B
    assert side.totale != str(len(side.circostanze))


def test_custom_case_follows_the_order():
    t, img = synth.custom(77, scenario="rotatoria", defects=["correzione"], quality="foto_scarsa", feriti=True,
                          received=date(2026, 11, 21))
    assert t.scenario == "rotatoria" and t.feriti and t.defects == ["correzione"]
    assert formats.parse_date(t.data) < date(2026, 11, 21)
    assert img.width > 500


def test_holdout_seeds_do_not_overlap_dev():
    assert not set(synth.seeds("dev", 200)) & set(synth.seeds("holdout", 100))
