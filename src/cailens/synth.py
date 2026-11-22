# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""Synthetic filled-in CAI forms with their ground truth.

``make_case(seed, split)`` decides what is written (people, plates, circumstances, defects) and
``render(truth)`` draws it on the blank form in block capitals, as the form asks, then degrades the
page like a scan or a phone photo. Everything is generated: no real person, plate, policy or insurer.
The holdout split uses a different seed range, harsher capture and two handwriting styles the dev
split never shows, so its numbers measure generalisation rather than memory.
"""

from __future__ import annotations

import io
import itertools
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from cailens import form
from cailens.form import Box, fields
from cailens.schema import CaseTruth, Party, Scenario, Zone

# --- synthetic people and places (fictional insurers on purpose) ------------------------------------
SURNAMES = ("ROSSI", "BIANCHI", "ROMANO", "COLOMBO", "RICCI", "MARINO", "GRECO", "BRUNO", "GALLO", "CONTI",
            "DE LUCA", "MANCINI", "COSTA", "GIORDANO", "RIZZO", "LOMBARDI", "MORETTI", "BARBIERI", "FONTANA",
            "SANTORO", "MARIANI", "RINALDI", "CARUSO", "FERRARI", "GALLI", "MARTINI", "LEONE", "LONGO",
            "GENTILE", "MARTINELLI", "VITALE", "SERRA", "COPPOLA", "DE SANTIS", "D'ANGELO", "MARCHETTI")
NAMES_M = ("MARCO", "LUCA", "GIUSEPPE", "ANDREA", "FRANCESCO", "ALESSANDRO", "MATTEO", "LORENZO", "DAVIDE",
           "STEFANO", "PAOLO", "ROBERTO", "GIOVANNI", "SIMONE", "FEDERICO", "NICOLA")
NAMES_F = ("GIULIA", "FRANCESCA", "SARA", "CHIARA", "ELENA", "VALENTINA", "LAURA", "ALESSIA", "MARTINA",
           "SILVIA", "ANNA", "ROBERTA", "FEDERICA", "PAOLA", "MONICA", "ELISA")
INSURERS = ("AURORA ASSICURAZIONI", "POLARIS ASSICURA", "MEDITERRANEA RC", "ALPINA MUTUA",
            "PONENTE ASSICURAZIONI", "LAGUNA ASS.NI", "BOREALE RC AUTO", "CORSO MUTUA")
TOWNS = (("MILANO", "F205"), ("ROMA", "H501"), ("TORINO", "L219"), ("NAPOLI", "F839"), ("BOLOGNA", "A944"),
         ("FIRENZE", "D612"), ("VENEZIA", "L736"), ("BERGAMO", "A794"), ("VERONA", "L781"), ("PADOVA", "G224"))
STREETS = ("VIA ROMA", "VIA GARIBALDI", "CORSO ITALIA", "VIA MAZZINI", "VIALE EUROPA", "VIA DANTE",
           "VIA VERDI", "PIAZZA CAVOUR", "VIA MARCONI", "VIA MILANO", "VIALE MONZA", "VIA TORINO")
DAMAGE = {
    "anteriore": ("PARAURTI ANTERIORE", "COFANO E FARO", "MASCHERINA"),
    "posteriore": ("PARAURTI POSTERIORE", "PORTELLONE", "FANALE POST."),
    "lato_sx": ("PORTIERA SX", "FIANCATA SINISTRA", "SPECCHIETTO SX"),
    "lato_dx": ("PORTIERA DX", "FIANCATA DESTRA", "SPECCHIETTO DX"),
    "ant_sx": ("FARO ANT. SX", "PARAFANGO ANT. SX"),
    "ant_dx": ("FARO ANT. DX", "PARAFANGO ANT. DX"),
    "post_sx": ("FANALE POST. SX", "PARAFANGO POST. SX"),
    "post_dx": ("FANALE POST. DX", "PARAFANGO POST. DX"),
}
NOTES = ("", "", "", "TRAFFICO INTENSO", "STRADA BAGNATA", "SEMAFORO GUASTO", "VISIBILITA' SCARSA")

# --- codice fiscale and plates: format-valid synthetic identifiers --------------------------------
_MONTHS = "ABCDEHLMPRST"
_ODD = dict(zip("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                (1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8,
                 12, 14, 16, 10, 22, 25, 24, 23), strict=True))
_EVEN = {**{str(d): d for d in range(10)}, **{chr(65 + i): i for i in range(26)}}
PLATE_LETTERS = "ABCDEFGHJKLMNPRSTVWXYZ"  # I, O, Q, U are not used on Italian plates


def cf_check(first15: str) -> str:
    total = sum(_ODD[c] if i % 2 == 0 else _EVEN[c] for i, c in enumerate(first15))
    return chr(65 + total % 26)


def _letters(word: str) -> tuple[str, str]:
    w = "".join(c for c in word.upper() if c.isalpha())
    return "".join(c for c in w if c not in "AEIOU"), "".join(c for c in w if c in "AEIOU")


def codice_fiscale(surname: str, name: str, born: date, female: bool, town: str) -> str:
    cons, vow = _letters(surname)
    s = (cons + vow + "XXX")[:3]
    cons, vow = _letters(name)
    n = (cons[0] + cons[2] + cons[3]) if len(cons) >= 4 else (cons + vow + "XXX")[:3]
    day = born.day + (40 if female else 0)
    body = f"{s}{n}{born.year % 100:02d}{_MONTHS[born.month - 1]}{day:02d}{town}"
    return body + cf_check(body)


def plate(rng: random.Random) -> str:
    L = PLATE_LETTERS
    return f"{rng.choice(L)}{rng.choice(L)}{rng.randint(0, 999):03d}{rng.choice(L)}{rng.choice(L)}"


# --- scenarios: circumstances and impact zones that belong together ---------------------------------
SCENARIO_RULES: dict[Scenario, tuple[list[int], list[int], Zone, Zone]] = {
    # A's boxes, B's boxes, A's impact zone, B's impact zone
    "tamponamento": ([8], [1], "anteriore", "posteriore"),
    "incrocio": ([17], [16], "anteriore", "lato_sx"),
    "cambio_corsia": ([10], [9], "ant_dx", "post_sx"),
    "parcheggio": ([4], [], "ant_sx", "lato_dx"),
    "retromarcia": ([14], [1], "posteriore", "anteriore"),
    "rotatoria": ([6], [7], "anteriore", "lato_dx"),
}
EXTRA = {"incrocio": (13, 12), "tamponamento": (9,), "rotatoria": (12,), "cambio_corsia": (11,)}


@dataclass(frozen=True)
class Profile:
    """How hard a split is."""

    photo_share: float
    max_rot: float
    max_persp: float
    blur: tuple[float, float]
    jpeg: tuple[int, int]
    hands: tuple[str, ...]
    seed_base: int
    fold: float = 0.3  # share of pages folded (a crease line crosses the boxes)
    dots: float = 0.15  # share of pages with stray pen dots in empty boxes


PROFILES = {
    "dev": Profile(0.55, 3.0, 0.035, (0.0, 0.9), (60, 92), ("geist.woff2",), 10_000),
    "holdout": Profile(0.75, 6.0, 0.05, (0.3, 1.3), (50, 85),
                       ("geist.woff2", "geist-mono.woff2", "ibm-plex-serif.woff2"), 90_000, 0.5, 0.25),
    "demo": Profile(0.5, 2.5, 0.03, (0.0, 0.8), (70, 92), ("geist.woff2", "geist-mono.woff2"), 500_000),
    "scarsa": Profile(1.0, 8.0, 0.06, (1.7, 2.2), (25, 40), ("geist.woff2",), 600_000),
}
DEFECTS = (  # (code, probability) - several can occur together
    ("firma_B_mancante", 0.14), ("firma_A_mancante", 0.05), ("totale_errato", 0.14), ("totale_mancante", 0.07),
    ("correzione", 0.10), ("segno_debole", 0.10), ("campo_B_mancante", 0.07), ("data_successiva", 0.04),
    ("cf_errato", 0.06), ("denuncia_unilaterale", 0.05), ("firma_sigla", 0.10),
)


def _person(rng: random.Random) -> tuple[str, str, str]:
    female = rng.random() < 0.45
    surname, name = rng.choice(SURNAMES), rng.choice(NAMES_F if female else NAMES_M)
    born = date(rng.randint(1950, 2004), rng.randint(1, 12), rng.randint(1, 28))
    return surname, name, codice_fiscale(surname, name, born, female, rng.choice(TOWNS)[1])


def _party(rng: random.Random, boxes: list[int], zone: Zone) -> Party:
    surname, name, cf = _person(rng)
    same_driver = rng.random() < 0.7
    d_surname, d_name, _ = (surname, name, cf) if same_driver else _person(rng)
    return Party(
        cognome=surname, nome=name, cf=cf, targa=plate(rng), compagnia=rng.choice(INSURERS),
        polizza=f"{rng.randint(10, 99)}{rng.choice('ABCDEFGHLMNPRSTV')}{rng.randint(1_000_000, 9_999_999)}",
        cond_cognome=d_surname, cond_nome=d_name,
        patente=f"{rng.choice('ABCDEMRTUV')}{rng.choice('ABCDEMRTUV')}{rng.randint(1_000_000, 9_999_999)}"
                f"{rng.choice('ABCDEFGHLMNPRSTVZ')}",
        danni=rng.choice(DAMAGE[zone]), osservazioni=rng.choice(NOTES) or None, impatto=zone,
        circostanze=sorted(boxes), totale=str(len(boxes)), firma=True,
    )


DEMO_ANCHOR = date(2026, 11, 24)  # the demo queue is received in the three weeks before this date


def make_case(seed: int, split: str = "dev", *, scenario: Scenario | None = None, defects: list[str] | None = None,
              received: date | None = None, feriti: bool | None = None) -> CaseTruth:
    """Decide everything that is written on one form (deterministic for a seed).

    The keyword arguments force a scenario, an exact list of defects, the date of receipt or the
    injuries box - used by the demo page that lets a visitor build a case."""
    rng = random.Random(seed)
    drawn: Scenario = rng.choice(tuple(SCENARIO_RULES))
    scenario = scenario or drawn
    a_boxes, b_boxes, za, zb = SCENARIO_RULES[scenario]
    a_boxes = list(a_boxes)
    if scenario in EXTRA and rng.random() < 0.3:
        a_boxes.append(rng.choice(EXTRA[scenario]))
    A, B = _party(rng, a_boxes, za), _party(rng, list(b_boxes), zb)
    offset = rng.randint(0, 320)
    if received is None:
        received = DEMO_ANCHOR - timedelta(days=offset % 21) if split == "demo" else date(2026, 1, 5) + timedelta(days=offset)
    when = received - timedelta(days=rng.randint(0, 20))
    town = rng.choice(TOWNS)[0]
    truth = CaseTruth(
        case_id=f"{split}-{seed}", split=split, seed=seed,
        data=when.strftime("%d/%m/%Y"), ora=f"{rng.randint(6, 22):02d}:{rng.choice((0, 10, 15, 20, 30, 40, 45, 50)):02d}",
        luogo=f"{town}, {rng.choice(STREETS)} {rng.randint(1, 180)}",
        feriti=rng.random() < 0.12, veicoli=not rng.random() < 0.9,
        oggetti=not rng.random() < 0.85,
        testimoni=None if rng.random() < 0.6 else f"{rng.choice(SURNAMES)} {rng.choice(NAMES_M + NAMES_F)}",
        received_on=received, scenario=scenario, A=A, B=B,
    )
    draws = [code for code, p in DEFECTS if rng.random() < p]
    truth.defects = list(defects) if defects is not None else draws
    if feriti is not None:
        truth.feriti = feriti
    _apply_defects(truth, rng)
    return truth


def _apply_defects(t: CaseTruth, rng: random.Random) -> None:
    d = set(t.defects)
    if "denuncia_unilaterale" in d:  # the other driver refused to sign: only plate and insurer of B
        t.B = Party(targa=t.B.targa, compagnia=t.B.compagnia, firma=False, totale=None)
        d -= {"firma_B_mancante", "campo_B_mancante"}
    if "firma_B_mancante" in d:
        t.B.firma = False
    if "firma_A_mancante" in d:
        t.A.firma = False
    side = t.A if rng.random() < 0.6 or "denuncia_unilaterale" in d else t.B
    if "totale_mancante" in d:
        side.totale = None
    elif "totale_errato" in d and side.totale is not None:
        n = int(side.totale)
        side.totale = str(n + 1 if n == 0 or rng.random() < 0.6 else n - 1)
    if "correzione" in d:
        free = [n for n in range(1, 18) if n not in side.circostanze]
        side.annullate = [rng.choice(free)]
    if "segno_debole" in d and not side.circostanze:
        d.discard("segno_debole")
    if "campo_B_mancante" in d:
        if rng.random() < 0.5:
            t.B.targa = None
        else:
            t.B.compagnia = None
    if "data_successiva" in d:
        t.data = (t.received_on + timedelta(days=rng.randint(2, 30))).strftime("%d/%m/%Y")
    if "cf_errato" in d and t.A.cf:
        cf = t.A.cf
        wrong = next(c for c in "ABCDEFGHLMNPRSTVZ" if c != cf[-1])
        t.A.cf = cf[:-1] + wrong
    t.defects = [c for c, _ in DEFECTS if c in d]


# --- drawing the handwriting --------------------------------------------------------------------
@dataclass
class Hand:
    font: str
    size: float  # glyph height as a share of the field height
    slant: float
    jitter: float
    ink: tuple[int, int, int]
    weight: int
    track: float


def hand_for(rng: random.Random, profile: Profile) -> Hand:
    inks = ((22, 38, 120), (20, 20, 28), (30, 52, 150), (40, 40, 60))
    return Hand(font=rng.choice(profile.hands), size=rng.uniform(0.62, 0.86), slant=rng.uniform(-0.18, 0.12),
                jitter=rng.uniform(0.6, 1.6), ink=rng.choice(inks), weight=rng.choice((0, 0, 1)),
                track=rng.uniform(0.0, 0.18))


def write(img: Image.Image, text: str, box: Box, hand: Hand, rng: random.Random) -> None:
    """Block capitals, one glyph at a time: size, angle, baseline and spacing all wander a little."""
    if not text:
        return
    size = max(12, int(box.h * hand.size))
    f = form.font(size, hand.font)
    need = f.getlength(text.upper()) * (1 + hand.track) + 14
    if need > box.w:  # people write smaller when the line is short
        size = max(11, int(size * box.w / need))
        f = form.font(size, hand.font)
    x = box.x0 + rng.randint(0, 8)
    base = box.y1 - rng.randint(1, 4)
    drift = 0.0
    for ch in text.upper():
        if ch == " ":
            x += int(size * rng.uniform(0.35, 0.55))
            continue
        drift = max(-4.0, min(4.0, drift + rng.uniform(-1, 1) * hand.jitter))
        s = max(10, int(size * rng.uniform(0.92, 1.08)))
        g = form.font(s, hand.font)
        bw = int(g.getlength(ch)) + 8
        patch = Image.new("L", (bw + s, int(s * 1.5)), 0)
        ImageDraw.Draw(patch).text((s // 2, s * 1.2), ch, fill=255, font=g, anchor="ls", stroke_width=hand.weight)
        shear = hand.slant + rng.uniform(-0.05, 0.05) * hand.jitter
        patch = patch.transform(patch.size, Image.Transform.AFFINE, (1, shear, -shear * patch.size[1] / 2, 0, 1, 0),
                                resample=Image.Resampling.BICUBIC)
        patch = patch.rotate(rng.uniform(-4, 4) * hand.jitter, resample=Image.Resampling.BICUBIC)
        top = int(base + drift - s * 1.2)
        if x + bw > box.x1 + 4:
            break  # people write smaller at the end, or run out of space: we just stop
        img.paste(Image.new("RGB", patch.size, hand.ink), (int(x - s // 2), top), patch)
        x += int(g.getlength(ch) * (1 + hand.track) + rng.uniform(-1, 2))
    del f


def _stroke(d: ImageDraw.ImageDraw, pts: list[tuple[float, float]], ink: tuple[int, int, int], w: int,
            rng: random.Random, wobble: float = 1.2) -> None:
    out = []
    for (x0, y0), (x1, y1) in itertools.pairwise(pts):
        steps = max(2, int(math.hypot(x1 - x0, y1 - y0) / 4))
        for i in range(steps):
            t = i / steps
            out.append((x0 + (x1 - x0) * t + rng.uniform(-wobble, wobble), y0 + (y1 - y0) * t + rng.uniform(-wobble, wobble)))
    out.append(pts[-1])
    d.line(out, fill=ink, width=w, joint="curve")


def mark(img: Image.Image, box: Box, hand: Hand, rng: random.Random, style: str | None = None) -> None:
    """A cross (as the form asks), a tick, or a filled box - strokes overshoot the borders a little."""
    d = ImageDraw.Draw(img)
    style = style or rng.choices(("x", "tick", "fill"), weights=(70, 22, 8), k=1)[0]
    w = rng.choice((2, 3, 3, 4))
    m = rng.uniform(-3, 4)
    x0, y0, x1, y1 = box.x0 + m, box.y0 + m, box.x1 - m, box.y1 - m
    if style == "x":
        _stroke(d, [(x0, y0), (x1, y1)], hand.ink, w, rng)
        _stroke(d, [(x1, y0 + rng.uniform(-2, 2)), (x0 + rng.uniform(-2, 2), y1)], hand.ink, w, rng)
    elif style == "tick":
        _stroke(d, [(x0, (y0 + y1) / 2), ((x0 + x1) / 2 - 2, y1), (x1 + 4, y0 - 4)], hand.ink, w, rng)
    elif style == "weak":  # a small, light cross: really meant, hard to see
        cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, box.w * rng.uniform(0.14, 0.22)
        light = tuple(min(255, c + 70) for c in hand.ink)
        _stroke(d, [(cx - r, cy - r), (cx + r, cy + r)], light, 2, rng, 0.4)  # type: ignore[arg-type]
        _stroke(d, [(cx + r, cy - r), (cx - r, cy + r)], light, 2, rng, 0.4)  # type: ignore[arg-type]
    elif style in ("fill", "scribble"):  # dense zig-zag; "scribble" = a cross scratched out
        n = 5 if style == "fill" else 14
        pts = []
        for i in range(n):
            t = i / (n - 1)
            pts.append((x0 + 1 if i % 2 == 0 else x1 - 1, y0 + 2 + t * (y1 - y0 - 4)))
        _stroke(d, pts, hand.ink, w + 1, rng, 0.8)
        if style == "scribble":
            pts = [(x0 + 2 + (i / 13) * (x1 - x0 - 4), y0 + 1 if i % 2 == 0 else y1 - 1) for i in range(14)]
            _stroke(d, pts, hand.ink, w + 1, rng, 0.8)


def _catmull(pts: list[tuple[float, float]], n: int = 8) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    p = [pts[0], *pts, pts[-1]]
    for i in range(len(p) - 3):
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = p[i:i + 4]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            def cr(a: float, b: float, c: float, e: float, t: float = t, t2: float = t2, t3: float = t3) -> float:
                return 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - e) * t2 + (-a + 3 * b - 3 * c + e) * t3)

            out.append((cr(x0, x1, x2, x3), cr(y0, y1, y2, y3)))
    out.append(pts[-1])
    return out


def signature(img: Image.Image, box: Box, hand: Hand, rng: random.Random, small: bool = False) -> None:
    """A pen line through random control points: a capital loop, a cursive run, a tail (or initials)."""
    d = ImageDraw.Draw(img)
    span = box.w * (rng.uniform(0.1, 0.16) if small else rng.uniform(0.38, 0.66))
    x = box.x0 + rng.uniform(0.08, 0.3) * box.w
    cy = box.y0 + box.h * rng.uniform(0.42, 0.62)
    amp = box.h * (0.14 if small else rng.uniform(0.16, 0.3))
    # capital: a tall loop
    pts = [(x, cy + amp * 0.6), (x + amp * 0.3, cy - amp * 1.6), (x + amp * 0.9, cy - amp * 0.4),
           (x + amp * 0.2, cy + amp * 0.2), (x + amp * 1.1, cy)]
    k = rng.randint(2, 3) if small else rng.randint(6, 12)
    cx = x + amp * 1.1
    for i in range(k):
        cx += span / k * rng.uniform(0.7, 1.3)
        up = i % 2 == 0
        h = amp * rng.uniform(0.3, 1.2) * (1 if up else -0.5)
        pts.append((cx, cy - h))
        if rng.random() < 0.15:  # a small loop
            pts += [(cx + 6, cy - h - 10), (cx - 4, cy - h - 6), (cx + 3, cy - h)]
    pts.append((cx + rng.uniform(10, 40), cy + amp * rng.uniform(0.2, 0.7)))
    d.line(_catmull(pts), fill=hand.ink, width=rng.choice((2, 2, 3)), joint="curve")
    if not small and rng.random() < 0.4:
        _stroke(d, [(x + amp, cy + amp * 1.2), (cx + 20, cy + amp * 1.0)], hand.ink, 2, rng)


ZONE_TARGET: dict[Zone, tuple[float, float, float, float]] = {
    # point on the car body (fractions of width/height) and the outward direction of the arrow
    "anteriore": (0.5, 0.0, 0, -1), "posteriore": (0.5, 1.0, 0, 1), "lato_sx": (0.0, 0.5, -1, 0),
    "lato_dx": (1.0, 0.5, 1, 0), "ant_sx": (0.05, 0.08, -0.7, -0.7), "ant_dx": (0.95, 0.08, 0.7, -0.7),
    "post_sx": (0.05, 0.92, -0.7, 0.7), "post_dx": (0.95, 0.92, 0.7, 0.7),
}


def impact_arrow(img: Image.Image, side: str, zone: Zone, hand: Hand, rng: random.Random) -> None:
    body = form.car_rect(form.IMPACT_BOX[side])
    fx, fy, dx, dy = ZONE_TARGET[zone]
    tx, ty = body.x0 + fx * body.w, body.y0 + fy * body.h
    norm = math.hypot(dx, dy)
    dx, dy = dx / norm, dy / norm
    length = rng.uniform(48, 70)
    sx, sy = tx + dx * (length + 8), ty + dy * (length + 8)
    ex, ey = tx + dx * 6, ty + dy * 6
    d = ImageDraw.Draw(img)
    _stroke(d, [(sx, sy), (ex, ey)], hand.ink, 3, rng)
    ang = math.atan2(ey - sy, ex - sx)
    for da in (2.6, -2.6):
        _stroke(d, [(ex, ey), (ex + 16 * math.cos(ang + da), ey + 16 * math.sin(ang + da))], hand.ink, 3, rng, 0.6)


def _car(d: ImageDraw.ImageDraw, cx: float, cy: float, angle: float, label: str, ink: tuple[int, int, int],
         rng: random.Random) -> None:
    """A small labelled rectangle with a direction arrow: how people draw cars in section 13."""
    w, h = 34, 62
    ca, sa = math.cos(angle), math.sin(angle)

    def rot(px: float, py: float) -> tuple[float, float]:
        return (cx + px * ca - py * sa, cy + px * sa + py * ca)

    corners = [rot(-w / 2, -h / 2), rot(w / 2, -h / 2), rot(w / 2, h / 2), rot(-w / 2, h / 2), rot(-w / 2, -h / 2)]
    _stroke(d, corners, ink, 2, rng, 0.8)
    tip = rot(0, -h / 2 - 40)
    _stroke(d, [rot(0, -h / 2 - 6), tip], ink, 2, rng, 0.6)
    _stroke(d, [rot(-8, -h / 2 - 30), tip, rot(8, -h / 2 - 30)], ink, 2, rng, 0.6)
    d.text(rot(0, 0), label, fill=ink, font=form.font(22), anchor="mm")


def sketch(img: Image.Image, scenario: Scenario, hand: Hand, rng: random.Random) -> None:
    b = form.SKETCH_BOX
    d = ImageDraw.Draw(img)
    cx, cy = (b.x0 + b.x1) / 2 + rng.uniform(-60, 60), (b.y0 + b.y1) / 2 + rng.uniform(-10, 10)
    road = (90, 90, 90)
    up, right, _down, left = 0.0, math.pi / 2, math.pi, -math.pi / 2
    if scenario in ("tamponamento", "cambio_corsia", "parcheggio", "retromarcia"):
        _stroke(d, [(b.x0 + 30, cy - 70), (b.x1 - 30, cy - 70)], road, 2, rng, 0.8)
        _stroke(d, [(b.x0 + 30, cy + 70), (b.x1 - 30, cy + 70)], road, 2, rng, 0.8)
        if scenario == "cambio_corsia":
            for x in range(b.x0 + 40, b.x1 - 40, 50):
                _stroke(d, [(x, cy), (x + 24, cy)], road, 2, rng, 0.4)
        if scenario == "tamponamento":
            _car(d, cx - 70, cy + 30, right, "A", hand.ink, rng)
            _car(d, cx + 4, cy + 30, right, "B", hand.ink, rng)
        elif scenario == "cambio_corsia":
            _car(d, cx - 40, cy - 30, right + 0.35, "A", hand.ink, rng)
            _car(d, cx + 40, cy + 34, right, "B", hand.ink, rng)
        elif scenario == "parcheggio":
            _car(d, cx - 20, cy + 95, up + 0.6, "A", hand.ink, rng)
            _car(d, cx + 40, cy + 20, right, "B", hand.ink, rng)
            d.text((cx - 140, cy + 105), "P", fill=hand.ink, font=form.font(26))
        else:  # retromarcia: A reverses (arrow drawn backwards) into B parked behind it
            _car(d, cx + 40, cy + 30, left, "A", hand.ink, rng)
            _car(d, cx - 40, cy + 30, right, "B", hand.ink, rng)
            d.text((cx + 80, cy - 50), "RETROMARCIA", fill=hand.ink, font=form.font(16))
    elif scenario == "incrocio":
        for off in (-60, 60):
            _stroke(d, [(b.x0 + 30, cy + off), (cx - 60, cy + off)], road, 2, rng, 0.8)
            _stroke(d, [(cx + 60, cy + off), (b.x1 - 30, cy + off)], road, 2, rng, 0.8)
            _stroke(d, [(cx + off, b.y0 + 10), (cx + off, cy - 60)], road, 2, rng, 0.8)
            _stroke(d, [(cx + off, cy + 60), (cx + off, b.y1 - 10)], road, 2, rng, 0.8)
        _car(d, cx - 70, cy + 25, right, "A", hand.ink, rng)
        _car(d, cx + 25, cy + 40, up, "B", hand.ink, rng)
        d.text((cx - 120, cy + 70), "STOP", fill=hand.ink, font=form.font(16))
    else:  # rotatoria
        d.ellipse((cx - 110, cy - 110, cx + 110, cy + 110), outline=road, width=2)
        d.ellipse((cx - 45, cy - 45, cx + 45, cy + 45), outline=road, width=2)
        _stroke(d, [(b.x0 + 40, cy - 30), (cx - 108, cy - 30)], road, 2, rng, 0.8)
        _stroke(d, [(b.x0 + 40, cy + 30), (cx - 108, cy + 30)], road, 2, rng, 0.8)
        _car(d, cx - 140, cy, right, "A", hand.ink, rng)
        _car(d, cx - 70, cy - 70, right + 0.6, "B", hand.ink, rng)


# --- the whole page ---------------------------------------------------------------------------------
def render_clean(t: CaseTruth) -> Image.Image:
    """The filled form before any capture effect: what the pen put on paper."""
    rng = random.Random(t.seed * 7 + 1)
    profile = PROFILES.get(t.split, PROFILES["dev"])
    img = form.blank().copy()
    F = fields()
    hand_a, hand_b = hand_for(rng, profile), hand_for(rng, profile)

    def put(fid: str, value: str | None, hand: Hand) -> None:
        if value:
            write(img, value, F[fid].box, hand, rng)

    put("data", t.data, hand_a)
    put("ora", t.ora, hand_a)
    put("luogo", t.luogo, hand_a)
    put("testimoni", t.testimoni or ("NESSUNO" if rng.random() < 0.5 else None), hand_a)
    for name, val in (("feriti", t.feriti), ("veicoli", t.veicoli), ("oggetti", t.oggetti)):
        if val is not None:
            mark(img, F[f"{name}_{'si' if val else 'no'}"].box, hand_a, rng)
    weak_done = False
    for side, party, hand in (("A", t.A, hand_a), ("B", t.B, hand_b)):
        for key in ("cognome", "nome", "cf", "targa", "compagnia", "polizza", "cond_cognome", "cond_nome",
                    "patente", "danni", "osservazioni"):
            put(f"{side}.{key}", getattr(party, key), hand)
        for n in party.circostanze:
            weak = "segno_debole" in t.defects and not weak_done and n == party.circostanze[0]
            weak_done = weak_done or weak
            mark(img, F[f"{side}.c{n:02d}"].box, hand, rng, "weak" if weak else None)
        for n in party.annullate:
            mark(img, F[f"{side}.c{n:02d}"].box, hand, rng, "scribble")
        if party.totale is not None:
            tb = F[f"{side}.totale"].box
            write(img, party.totale, Box(tb.x0 + 14, tb.y0 + 6, tb.x1 - 4, tb.y1 - 8), hand, rng)
        if party.impatto:
            impact_arrow(img, side, party.impatto, hand, rng)
        if party.firma:
            signature(img, F[f"{side}.firma"].box, hand, rng, small="firma_sigla" in t.defects and side == "B")
        elif rng.random() < 0.3:  # an empty signature box is rarely perfectly clean
            sb = F[f"{side}.firma"].box
            ImageDraw.Draw(img).ellipse((sb.x0 + 40, sb.y0 + 50, sb.x0 + 43, sb.y0 + 53), fill=hand.ink)
    if rng.random() < profile.dots:  # the pen rested on the paper: dots in empty boxes
        d = ImageDraw.Draw(img)
        exp_marked = {f"{sd}.c{n:02d}" for sd, p in (("A", t.A), ("B", t.B)) for n in p.circostanze + p.annullate}
        empty = [f for f in form.CHECKBOX_FIELDS if f.startswith(("A.", "B.")) and f not in exp_marked]
        for fid in rng.sample(empty, k=min(len(empty), rng.randint(1, 3))):
            b = F[fid].box
            x, y = rng.uniform(b.x0 + 6, b.x1 - 9), rng.uniform(b.y0 + 6, b.y1 - 9)
            r = rng.uniform(1.0, 2.0)
            d.ellipse((x - r, y - r, x + r, y + r), fill=hand_a.ink)
    if t.A.firma or t.B.firma:
        sketch(img, t.scenario, hand_a, rng)
    if rng.random() < profile.fold:  # folded in the glovebox: a grey crease across the page
        d = ImageDraw.Draw(img)
        y = rng.uniform(300, 1100)
        tilt = rng.uniform(-6, 6)
        d.line((0, y, form.W, y + tilt), fill=(150, 150, 155), width=2)
    return img


def capture(img: Image.Image, t: CaseTruth) -> Image.Image:
    """Scan or phone photo: rotation, perspective, a table behind the page, shadow, blur, noise, JPEG."""
    rng = random.Random(t.seed * 13 + 5)
    profile = PROFILES.get(t.split, PROFILES["dev"])
    if t.capture == "scansione":
        out = img.rotate(rng.uniform(-1.5, 1.5), resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
        scale = rng.uniform(0.85, 1.0)
    else:
        pad = int(W_PAD * rng.uniform(0.6, 1.0))
        bg = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad),
                       tuple(rng.randint(60, 120) for _ in range(3)))
        noise = np.random.default_rng(t.seed).normal(0, 10, (bg.height, bg.width, 1))
        bg = Image.fromarray(np.clip(np.asarray(bg, dtype=np.float32) + noise, 0, 255).astype(np.uint8))
        bg.paste(img, (pad, pad))
        bg = bg.rotate(rng.uniform(-profile.max_rot, profile.max_rot), resample=Image.Resampling.BICUBIC, expand=True,
                       fillcolor=bg.getpixel((2, 2)))
        w, h = bg.size
        j = profile.max_persp
        quad = [(rng.uniform(0, j) * w, rng.uniform(0, j) * h), (rng.uniform(0, j) * w, h - rng.uniform(0, j) * h),
                (w - rng.uniform(0, j) * w, h - rng.uniform(0, j) * h), (w - rng.uniform(0, j) * w, rng.uniform(0, j) * h)]
        # a true perspective (a phone photo is a homography), not a bilinear warp
        import cv2  # local import: only capture needs it

        m = cv2.getPerspectiveTransform(np.array([(0, 0), (0, h), (w, h), (w, 0)], np.float32), np.array(quad, np.float32))
        coeffs = (m.flatten() / m[2, 2])[:8]
        out = bg.transform((w, h), Image.Transform.PERSPECTIVE, tuple(float(c) for c in coeffs),
                           resample=Image.Resampling.BICUBIC, fillcolor=bg.getpixel((2, 2)))
        # uneven light: a soft gradient from one corner
        grad = np.linspace(rng.uniform(0.82, 0.95), 1.0, w, dtype=np.float32)[None, :, None]
        if rng.random() < 0.5:
            grad = grad[:, ::-1]
        out = Image.fromarray(np.clip(np.asarray(out, dtype=np.float32) * grad, 0, 255).astype(np.uint8))
        scale = rng.uniform(0.7, 0.95)
    out = out.resize((int(out.width * scale), int(out.height * scale)), Image.Resampling.LANCZOS)
    blur = rng.uniform(*profile.blur)
    if blur > 0.05:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(out, dtype=np.float32)
    arr += np.random.default_rng(t.seed + 1).normal(0, rng.uniform(1.5, 5.0), arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=rng.randint(*profile.jpeg))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


W_PAD = 110


def generate(seed: int, split: str = "dev") -> tuple[CaseTruth, Image.Image]:
    truth = make_case(seed, split)
    rng = random.Random(seed * 3 + 2)
    truth.capture = "foto" if rng.random() < PROFILES.get(split, PROFILES["dev"]).photo_share else "scansione"
    return truth, capture(render_clean(truth), truth)


CAPTURES = {"scansione": "scansione", "foto": "foto", "foto_scarsa": "foto"}


def custom(seed: int, *, scenario: Scenario | None, defects: list[str], quality: str, feriti: bool,
           received: date) -> tuple[CaseTruth, Image.Image]:
    """A case built to order for the demo: chosen story, chosen defects, chosen capture quality."""
    truth = make_case(seed, "demo", scenario=scenario, defects=defects, received=received, feriti=feriti)
    truth.data = (received - timedelta(days=random.Random(seed).randint(1, 6))).strftime("%d/%m/%Y")
    if "data_successiva" in defects:
        truth.data = (received + timedelta(days=5)).strftime("%d/%m/%Y")
    truth.capture = CAPTURES.get(quality, "foto")  # type: ignore[assignment]
    clean = render_clean(truth)
    if quality == "foto_scarsa":
        truth.split = "scarsa"
    img = capture(clean, truth)
    truth.split = "demo"
    return truth, img


def jpeg_bytes(img: Image.Image, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def seeds(split: str, n: int) -> list[int]:
    base = PROFILES[split].seed_base
    return [base + i for i in range(n)]
