"""Ink measurements on the dev split, by true label: the evidence behind the thresholds."""
import sys
import time
from collections import defaultdict

import numpy as np

from cailens import form
from cailens.agents import checkbox, registrar, signature
from cailens.synth import generate, seeds
from cailens.truth import expected

n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
by_label = defaultdict(list)
sig = defaultdict(list)
scores, t0 = [], time.time()
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "dev"
for s in seeds(SPLIT, n):
    t, img = generate(s, SPLIT)
    reg = registrar.register(img)
    scores.append((reg.score, t.capture))
    exp = expected(t)
    for fid, r in checkbox.ratios(reg).items():
        by_label[exp[fid]].append(r)
    for side in "AB":
        px, w = signature.measure(reg, form.fields()[f"{side}.firma"].box)
        small = "firma_sigla" in t.defects and side == "B" and t.B.firma
        sig[("sigla" if small else exp[f"{side}.firma"])].append((px, w))
print(f"{n} docs in {time.time() - t0:.1f}s")
print("registration score min/median:", round(min(s for s, _ in scores), 3), round(float(np.median([s for s, _ in scores])), 3))
for label, vals in by_label.items():
    v = np.array(vals)
    print(f"checkbox {label:10s} n={len(v):5d} min={v.min():.3f} p01={np.quantile(v, .01):.3f} p50={np.median(v):.3f} p99={np.quantile(v, .99):.3f} max={v.max():.3f}")
for label, vals in sig.items():
    px = np.array([p for p, _ in vals])
    w = np.array([w for _, w in vals])
    print(f"signature {label:6s} n={len(px)} px min={px.min()} p50={int(np.median(px))} max={px.max()} | width min={w.min()} max={w.max()}")
