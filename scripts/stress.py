"""Robustness curve: make the photos worse step by step and count two things per level -
documents the registrar refuses (safe: they go to a person) and SILENT errors (a mark or a
signature read wrong and not flagged as uncertain). The second number is the one that must stay 0."""
import json
import sys
from dataclasses import replace

from cailens import synth
from cailens.agents import checkbox, registrar, signature
from cailens.paths import HOME
from cailens.truth import AMBIGUOUS, expected

N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
LEVELS = [  # (name, rotation deg, perspective, blur sigma, jpeg quality, scale)
    ("buona", 3, 0.03, (0.3, 0.6), (80, 90), None),
    ("media", 6, 0.05, (0.8, 1.2), (55, 70), None),
    ("scarsa", 9, 0.07, (1.4, 1.8), (35, 50), None),
    ("pessima", 12, 0.09, (2.0, 2.6), (20, 30), None),
    ("estrema", 15, 0.12, (2.8, 3.4), (10, 18), None),
]
base = synth.PROFILES["holdout"]
out = []
for name, rot, persp, blur, jpeg, _ in LEVELS:
    synth.PROFILES["stress"] = replace(base, photo_share=1.0, max_rot=rot, max_persp=persp, blur=blur, jpeg=jpeg,
                                       seed_base=700_000)
    refused = flagged = silent = silent_flagged = boxes = sig_unsure = box_unsure = 0
    for i in range(N):
        t, img = synth.generate(700_000 + i, "stress")
        reg = registrar.register(img)
        if not reg.ok:
            refused += 1
            continue
        exp = expected(t)
        gate = not reg.quality_ok
        flagged += gate
        errs = 0
        for fid, r in checkbox.read(reg).items():
            boxes += 1
            if r.value == "incerto":
                box_unsure += exp[fid] != AMBIGUOUS
            elif exp[fid] != AMBIGUOUS and r.value != exp[fid]:
                errs += 1
        for fid, r in signature.read(reg).items():
            if r.value == "incerto":
                sig_unsure += 1
            elif r.value != exp[fid]:
                errs += 1
        if gate:
            silent_flagged += errs  # wrong, but the document is in review with "check every box"
        else:
            silent += errs
    row = {"livello": name, "rotazione_max": rot, "sfocatura": blur, "jpeg": jpeg, "documenti": N,
           "rifiutati_dal_registrar": refused, "segnalati_qualita_bassa": flagged,
           "errori_silenziosi": silent, "errori_in_documenti_segnalati": silent_flagged, "caselle_lette": boxes,
           "caselle_mandate_a_revisione": box_unsure, "firme_incerte": sig_unsure}
    print(row, flush=True)
    out.append(row)
(HOME / "evals" / "results" / "stress.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
