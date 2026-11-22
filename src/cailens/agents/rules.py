# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Validator, deadline and router agents: plain code, no model.

- ``validate`` checks the form against itself: the declared number of crosses, the two signatures, the
  data the instructions make mandatory for vehicle B, the checksum of the codice fiscale, dates, plates,
  and whether the sketch tells the same story as the boxes.
- ``deadline`` applies art. 148 CAP: 60 days for damage to things, 30 when both drivers signed the form,
  90 with injuries. When an input is uncertain it takes the SHORTEST term the facts allow and marks it
  provisional - an early reminder costs nothing, a missed legal term does. Suspending the offer for
  suspected fraud (c. 2-bis) is never decided here.
- ``route`` says whether the extracted data can go to the liquidator as they are, or which fields a
  person must look at first. It never decides liability or an offer.
"""

from __future__ import annotations

from datetime import date, timedelta

from cailens import form, formats
from cailens.schema import Deadline, Issue, Reading

TEXT_MIN = 0.8  # below this a critical text field goes to review
CRITICAL_TEXT = ("data", "A.targa", "B.targa", "A.compagnia", "B.compagnia", "A.polizza", "A.cf", "A.cognome",
                 "A.totale", "B.totale")
SCENARIO_BOXES = {  # the box that tells each story, on either side
    "tamponamento": {8}, "incrocio": {16, 17}, "cambio_corsia": {10}, "parcheggio": {4, 5},
    "retromarcia": {14}, "rotatoria": {6, 7},
}


def _v(readings: dict[str, Reading], fid: str) -> str | None:
    r = readings.get(fid)
    return r.value if r else None


def marked(readings: dict[str, Reading], side: str) -> tuple[set[int], set[int]]:
    """Boxes read as crossed, and boxes nobody can be sure about, for one vehicle."""
    yes, unsure = set(), set()
    for n in range(1, 18):
        v = _v(readings, f"{side}.c{n:02d}")
        if v == "si":
            yes.add(n)
        elif v == "incerto":
            unsure.add(n)
    return yes, unsure


def validate(readings: dict[str, Reading], received_on: date, text_ok: bool = True) -> list[Issue]:
    """``text_ok=False`` (no handwriting reading) skips every check that needs a text field."""
    out: list[Issue] = []
    add = out.append
    for side in form.SIDES if text_ok else ():
        yes, unsure = marked(readings, side)
        total = _v(readings, f"{side}.totale")
        if total is not None and formats.shape_ok(f"{side}.totale", total):
            if unsure:
                add(Issue(code=f"totale_{side}_non_verificabile", severity="attenzione",
                          message=f"Veicolo {side}: caselle incerte, il totale dichiarato ({total}) non si può verificare.",
                          fields=[f"{side}.totale"]))
            elif int(total) != len(yes):
                add(Issue(code=f"totale_{side}_non_coincide", severity="errore",
                          message=f"Veicolo {side}: dichiarate {total} caselle, ne risultano barrate {len(yes)}.",
                          fields=[f"{side}.totale", *(f"{side}.c{n:02d}" for n in sorted(yes))]))
        elif total is None and yes:
            add(Issue(code=f"totale_{side}_mancante", severity="attenzione",
                      message=f"Veicolo {side}: numero di caselle barrate non indicato.", fields=[f"{side}.totale"]))

    sig = {s: _v(readings, f"{s}.firma") for s in form.SIDES}
    if "incerto" in sig.values():
        add(Issue(code="firma_incerta", severity="attenzione", message="Presenza di una firma da verificare: "
                  "decide tra 30 e 60 giorni per l'offerta.", fields=[f"{s}.firma" for s, v in sig.items() if v == "incerto"]))
    elif list(sig.values()).count("si") < 2:
        missing = [s for s, v in sig.items() if v != "si"]
        add(Issue(code="firma_mancante", severity="attenzione",
                  message=f"Manca la firma del conducente {', '.join(missing)}: termine di 60 giorni, non 30.",
                  fields=[f"{s}.firma" for s in missing]))

    f_si, f_no = _v(readings, "feriti_si"), _v(readings, "feriti_no")
    if f_si == "si" and f_no == "si":
        add(Issue(code="feriti_contraddittorio", severity="errore", message="Feriti: barrati sia «no» sia «sì».",
                  fields=["feriti_si", "feriti_no"]))
    elif f_si != "si" and f_no != "si":
        add(Issue(code="feriti_non_indicato", severity="attenzione",
                  message="Feriti: nessuna casella barrata o casella incerta.", fields=["feriti_si", "feriti_no"]))

    if not text_ok:
        return out
    when = formats.parse_date(_v(readings, "data"))
    if _v(readings, "data") and when is None:
        add(Issue(code="data_non_valida", severity="attenzione", message="Data dell'incidente non leggibile come data.",
                  fields=["data"]))
    elif when and when > received_on:
        add(Issue(code="data_successiva", severity="errore",
                  message=f"Data dell'incidente ({when:%d/%m/%Y}) successiva alla ricezione ({received_on:%d/%m/%Y}).",
                  fields=["data"]))

    for fid in ("B.targa", "B.compagnia"):
        if not _v(readings, fid):
            add(Issue(code="dati_B_mancanti", severity="errore",
                      message="Mancano targa o compagnia del veicolo B (obbligatori anche se B non firma).", fields=[fid]))
    ta, tb = formats.compact("A.targa", _v(readings, "A.targa")), formats.compact("B.targa", _v(readings, "B.targa"))
    if ta and tb and ta == tb:
        add(Issue(code="targhe_uguali", severity="errore", message="Stessa targa per A e B.", fields=["A.targa", "B.targa"]))
    for side in form.SIDES:
        fid = f"{side}.targa"
        if _v(readings, fid) and formats.shape_ok(fid, _v(readings, fid)) is False:
            add(Issue(code="targa_formato", severity="attenzione",
                      message=f"Targa {side} non nel formato italiano (AA000AA): estera o letta male.", fields=[fid]))
        fid = f"{side}.cf"
        if _v(readings, fid) and not formats.cf_valid(_v(readings, fid)):
            add(Issue(code="cf_non_valido", severity="errore",
                      message=f"Codice fiscale {side}: il carattere di controllo non torna (scritto o letto male).",
                      fields=[fid]))

    if _v(readings, "B.firma") == "no" and not _v(readings, "B.cognome"):
        add(Issue(code="denuncia_unilaterale", severity="attenzione",
                  message="Denuncia compilata solo dal conducente A: non vale come constatazione congiunta.",
                  fields=["B.firma"]))

    scen = readings.get("scenario")
    if scen and scen.value in SCENARIO_BOXES and scen.confidence >= 0.6:
        boxes = marked(readings, "A")[0] | marked(readings, "B")[0]
        if boxes and not boxes & SCENARIO_BOXES[scen.value]:
            add(Issue(code="grafico_incoerente", severity="attenzione",
                      message=f"Il grafico sembra un {scen.value}, ma nessuna casella corrispondente è barrata.",
                      fields=["schizzo"]))
    return out


def deadline(readings: dict[str, Reading], received_on: date) -> Deadline:
    sig = [_v(readings, f"{s}.firma") for s in form.SIDES]
    injured, f_no = _v(readings, "feriti_si"), _v(readings, "feriti_no")
    note = " La sospensione per sospetta frode (c. 2-bis) è sempre decisione del liquidatore."
    if injured == "si" and f_no != "si":
        return Deadline(days=90, due=received_on + timedelta(days=90), provisional=False,
                        basis="Lesioni alla persona: 90 giorni dalla ricezione (art. 148 c. 2 CAP)." + note)
    unsure = "incerto" in sig or injured == "incerto" or (injured == f_no)
    if all(v == "si" for v in sig):
        days, basis = 30, "Modulo firmato da entrambi i conducenti: 30 giorni (art. 148 c. 1 CAP)."
    elif "incerto" in sig:
        days, basis = 30, ("Firma incerta: si applica in via prudenziale il termine più breve, 30 giorni, "
                           "finché una persona non verifica (art. 148 c. 1 CAP).")
    else:
        days, basis = 60, "Danni a cose, modulo senza la firma di entrambi: 60 giorni (art. 148 c. 1 CAP)."
    if injured != "si" and injured == f_no:
        basis += " Casella «feriti» non chiara: con lesioni il termine sarebbe di 90 giorni."
    return Deadline(days=days, due=received_on + timedelta(days=days), provisional=unsure, basis=basis + note)


def route(readings: dict[str, Reading], issues: list[Issue], dl: Deadline, registered: bool,
          handwriting_ok: bool, quality_note: str = "") -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not registered:
        return "revisione", ["Modulo non allineato in modo affidabile: lettura automatica non attendibile."]
    if quality_note:
        reasons.append(f"Foto di qualità bassa ({quality_note}): verificare a vista tutte le caselle e le firme.")
    if not handwriting_ok:
        reasons.append("Lettura della scrittura a mano non disponibile: i campi di testo vanno inseriti a mano.")
    unsure_boxes = [f for f in form.CHECKBOX_FIELDS if _v(readings, f) == "incerto"]
    if unsure_boxes:
        reasons.append(f"Caselle incerte: {', '.join(unsure_boxes)} (forse una croce annullata).")
    for s in form.SIDES:
        if _v(readings, f"{s}.firma") == "incerto":
            reasons.append(f"Firma {s} da verificare.")
    if handwriting_ok:
        weak = [f for f in CRITICAL_TEXT if (r := readings.get(f)) and r.confidence < TEXT_MIN and
                not (r.value is None and r.confidence >= TEXT_MIN)]
        if weak:
            reasons.append(f"Campi chiave poco leggibili: {', '.join(weak)}.")
    for i in issues:
        if i.severity == "errore":
            reasons.append(i.message)
    if _v(readings, "feriti_si") == "si":
        reasons.append("Ci sono feriti: servono i documenti medici, gestione da parte di un liquidatore.")
    if dl.provisional:
        reasons.append("Termine dell'offerta provvisorio finché una persona non conferma firme e feriti.")
    return ("revisione" if reasons else "automatico"), reasons
