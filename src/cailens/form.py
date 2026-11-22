# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mykhailo Zvir
"""The front page of the CAI form: where every field is, and a renderer for the blank page.

The layout follows the structure of the Italian «Constatazione amichevole di incidente - Denuncia di
sinistro» (Reg. IVASS n. 56/2025, Allegato 1; the European accident statement): header boxes 1-5,
columns A and B with sections 6-11 and 14, the 17 circumstances of section 12 with a box per vehicle and
the declared number of marked boxes, the sketch (13) and the two signatures (15). It is drawn from
scratch here - no insurer's printed form is copied - and every page carries the words «fac-simile
dimostrativo». Coordinates are pixels of an A4 page at 150 dpi.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from cailens.paths import FONTS

W, H = 1240, 1754
FRAME = (40, 96, 1200, 1714)  # the thick outer border the registrar locks on to
FRAME_WIDTH = 4

Kind = Literal["text", "checkbox", "digit", "signature", "impact", "sketch"]
SIDES = ("A", "B")

CIRCUMSTANCES = (
    "in fermata / in sosta",
    "ripartiva dopo una sosta / apriva una portiera",
    "stava parcheggiando",
    "usciva da un parcheggio, da un luogo privato, da una strada vicinale",
    "entrava in un parcheggio, in un luogo privato, in una strada vicinale",
    "si immetteva in una piazza a senso rotatorio",
    "circolava su una piazza a senso rotatorio",
    "tamponava procedendo nello stesso senso e nella stessa fila",
    "procedeva nello stesso senso, ma in una fila diversa",
    "cambiava fila",
    "sorpassava",
    "girava a destra",
    "girava a sinistra",
    "retrocedeva",
    "invadeva la sede stradale riservata alla circolazione in senso inverso",
    "proveniva da destra",
    "non aveva osservato il segnale di precedenza o di semaforo rosso",
)

# colours of the printed form (the handwriting is drawn later, in synth.py)
PRINT = (38, 46, 60)
RULE = (120, 128, 140)
TINT = {"A": (232, 242, 251), "B": (255, 248, 214)}
HEAD = {"A": (44, 132, 196), "B": (230, 190, 20)}


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def inset(self, d: int) -> Box:
        return Box(self.x0 + d, self.y0 + d, self.x1 - d, self.y1 - d)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class Field:
    id: str
    kind: Kind
    box: Box  # where the writing goes (for text: the area above the rule)
    label: str
    side: str | None = None


# --- layout -------------------------------------------------------------------------------------
COL = {"A": (40, 470), "B": (770, 1200)}
CENTER = (470, 770)
ROW0 = 312  # first circumstance row
ROW_H = 42
CHECK = 26  # checkbox size
TEXT_ROWS = (  # (field suffix, label, y of the rule)
    ("cognome", "Cognome (stampatello)", 300),
    ("nome", "Nome", 338),
    ("cf", "Cod. Fisc. / P.IVA", 376),
    ("targa", "N. di targa", 440),
    ("compagnia", "Denominazione", 504),
    ("polizza", "N. di polizza", 542),
    ("cond_cognome", "Cognome", 606),
    ("cond_nome", "Nome", 644),
    ("patente", "Patente n.", 682),
)
SECTIONS = (  # (number, title, y) printed above the text rows of each column
    ("6", "contraente / assicurato", 254),
    ("7", "veicolo", 398),
    ("8", "compagnia d'assicurazione", 460),
    ("9", "conducente", 562),
)
IMPACT_BOX = {"A": Box(120, 735, 390, 975), "B": Box(850, 735, 1120, 975)}
SIGN_BOX = {"A": Box(90, 1572, 600, 1700), "B": Box(680, 1572, 1180, 1700)}
SKETCH_BOX = Box(60, 1190, 1180, 1514)
TOTAL_BOX = {"A": Box(478, ROW0 + 17 * ROW_H + 26, 538, ROW0 + 17 * ROW_H + 80),
             "B": Box(702, ROW0 + 17 * ROW_H + 26, 762, ROW0 + 17 * ROW_H + 80)}


def check_box(side: str, n: int) -> Box:
    """Box of circumstance ``n`` (1-17) for vehicle ``side``."""
    y = ROW0 + (n - 1) * ROW_H + (ROW_H - CHECK) // 2
    x = 480 if side == "A" else 734
    return Box(x, y, x + CHECK, y + CHECK)


def _text_box(x0: int, x1: int, y_rule: int, label_w: int) -> Box:
    return Box(x0 + label_w, y_rule - 28, x1 - 8, y_rule - 2)


@cache
def fields() -> dict[str, Field]:
    out: dict[str, Field] = {}

    def add(f: Field) -> None:
        out[f.id] = f

    add(Field("data", "text", Box(110, 128, 250, 158), "1. data"))
    add(Field("ora", "text", Box(296, 128, 374, 158), "ora"))
    add(Field("luogo", "text", Box(470, 128, 890, 158), "2. luogo"))
    add(Field("feriti_no", "checkbox", Box(1010, 130, 1036, 156), "3. feriti: no"))
    add(Field("feriti_si", "checkbox", Box(1110, 130, 1136, 156), "3. feriti: sì"))
    add(Field("veicoli_no", "checkbox", Box(260, 196, 286, 222), "4. danni a veicoli oltre A e B: no"))
    add(Field("veicoli_si", "checkbox", Box(340, 196, 366, 222), "4. danni a veicoli oltre A e B: sì"))
    add(Field("oggetti_no", "checkbox", Box(460, 196, 486, 222), "4. danni a oggetti: no"))
    add(Field("oggetti_si", "checkbox", Box(530, 196, 556, 222), "4. danni a oggetti: sì"))
    add(Field("testimoni", "text", Box(720, 192, 1190, 222), "5. testimoni"))
    for s in SIDES:
        x0, x1 = COL[s]
        for suffix, label, y in TEXT_ROWS:
            lw = 160 if suffix in ("cognome", "cf", "compagnia", "polizza") else 120
            add(Field(f"{s}.{suffix}", "text", _text_box(x0, x1, y, lw), label, s))
        add(Field(f"{s}.impatto", "impact", IMPACT_BOX[s], "10. punto d'urto iniziale", s))
        add(Field(f"{s}.danni", "text", _text_box(x0, x1, 1030, 10), "11. danni visibili", s))
        add(Field(f"{s}.osservazioni", "text", _text_box(x0, x1, 1110, 10), "14. osservazioni", s))
        for n in range(1, 18):
            add(Field(f"{s}.c{n:02d}", "checkbox", check_box(s, n), f"12.{n} {CIRCUMSTANCES[n - 1]}", s))
        add(Field(f"{s}.totale", "digit", TOTAL_BOX[s], "12. numero di caselle barrate", s))
        add(Field(f"{s}.firma", "signature", SIGN_BOX[s], "15. firma del conducente", s))
    add(Field("schizzo", "sketch", SKETCH_BOX, "13. grafico dell'incidente"))
    return out


TEXT_FIELDS = tuple(f.id for f in fields().values() if f.kind in ("text", "digit"))
CHECKBOX_FIELDS = tuple(f.id for f in fields().values() if f.kind == "checkbox")


# --- drawing --------------------------------------------------------------------------------------
@cache
def font(size: int, name: str = "geist.woff2") -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


def _wrap(text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if f.getlength(trial) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    return [*lines, cur] if cur else lines


def car_outline(d: ImageDraw.ImageDraw, box: Box, color: tuple[int, int, int] = RULE) -> None:
    """Top view of a car centred in ``box``: body, windscreen, rear window, wheels."""
    cx = (box.x0 + box.x1) // 2
    bw, bh = 96, 196
    x0, y0 = cx - bw // 2, box.y0 + (box.h - bh) // 2
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=26, outline=color, width=3)
    d.rounded_rectangle((x0 + 12, y0 + 40, x0 + bw - 12, y0 + 70), radius=8, outline=color, width=2)
    d.rounded_rectangle((x0 + 14, y0 + bh - 56, x0 + bw - 14, y0 + bh - 32), radius=6, outline=color, width=2)
    for wy in (y0 + 30, y0 + bh - 52):
        d.rectangle((x0 - 8, wy, x0, wy + 26), fill=color)
        d.rectangle((x0 + bw, wy, x0 + bw + 8, wy + 26), fill=color)
    d.text((cx, y0 - 8), "fronte", fill=RULE, font=font(13), anchor="md")


def car_rect(box: Box) -> Box:
    """The body rectangle drawn by ``car_outline`` (used to place impact arrows)."""
    cx = (box.x0 + box.x1) // 2
    bw, bh = 96, 196
    x0, y0 = cx - bw // 2, box.y0 + (box.h - bh) // 2
    return Box(x0, y0, x0 + bw, y0 + bh)


@cache
def blank() -> Image.Image:
    """The empty printed form (RGB). Cached: every synthetic document starts from a copy."""
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title, f_lab, f_small, f_sec = font(26), font(16), font(13), font(15)

    d.text((40, 30), "Constatazione amichevole di incidente - Denuncia di sinistro", fill=PRINT, font=f_title)
    d.text((40, 66), "Fac-simile dimostrativo · struttura del modulo di cui al Reg. IVASS n. 56/2025, "
           "Allegato 1 · dati sintetici", fill=RULE, font=f_small)
    d.text((1200, 66), "CAI-Lens", fill=RULE, font=f_small, anchor="ra")

    # column tints first, so the rules sit on top
    for s in SIDES:
        x0, x1 = COL[s]
        d.rectangle((x0, 232, x1, 1130), fill=TINT[s])
        d.rectangle((x0, 232, x1, 252), fill=HEAD[s])
        d.text(((x0 + x1) // 2, 242), f"veicolo {s}", fill="white" if s == "A" else PRINT, font=f_sec, anchor="mm")

    # header boxes 1-5
    d.line((40, 168, 1200, 168), fill=PRINT, width=2)
    d.line((40, 232, 1200, 232), fill=PRINT, width=2)
    for x in (380, 900):
        d.line((x, 96, x, 168), fill=PRINT, width=2)
    d.line((640, 168, 640, 232), fill=PRINT, width=2)
    d.text((50, 104), "1. data", fill=PRINT, font=f_lab)
    d.text((262, 104), "ora", fill=PRINT, font=f_lab)
    d.text((390, 104), "2. luogo (comune, via e numero)", fill=PRINT, font=f_lab)
    d.text((910, 104), "3. feriti anche se lievi", fill=PRINT, font=f_lab)
    d.text((50, 174), "4. danni materiali", fill=PRINT, font=f_lab)
    d.text((50, 200), "veicoli oltre A e B", fill=PRINT, font=f_small)
    d.text((392, 200), "oggetti", fill=PRINT, font=f_small)
    d.text((650, 174), "5. testimoni (cognome, nome, indirizzo)", fill=PRINT, font=f_lab)
    for name in ("feriti_no", "feriti_si", "veicoli_no", "veicoli_si", "oggetti_no", "oggetti_si"):
        b = fields()[name].box
        word = "no" if name.endswith("_no") else "sì"
        d.rectangle(b.as_tuple(), outline=PRINT, width=2)
        d.text((b.x0 - 6, (b.y0 + b.y1) // 2), word, fill=PRINT, font=f_small, anchor="rm")
    for name in ("data", "ora", "luogo"):
        b = fields()[name].box
        d.line((b.x0, b.y1 + 2, b.x1, b.y1 + 2), fill=RULE, width=1)
    b = fields()["testimoni"].box
    d.line((b.x0 - 70, b.y1 + 2, b.x1, b.y1 + 2), fill=RULE, width=1)

    # columns A and B: sections 6-9 with ruled lines
    for s in SIDES:
        x0, x1 = COL[s]
        for num, title, y in SECTIONS:
            d.text((x0 + 8, y), f"{num}. {title}", fill=PRINT, font=f_sec)
        for suffix, label, y in TEXT_ROWS:
            box = fields()[f"{s}.{suffix}"].box
            d.text((x0 + 8, y - 4), label, fill=PRINT, font=f_small, anchor="ls")
            d.line((box.x0, y, x1 - 8, y), fill=RULE, width=1)
        d.text((x0 + 8, 712), "10. indicare con una freccia il punto d'urto iniziale", fill=PRINT, font=f_sec)
        car_outline(d, IMPACT_BOX[s])
        for label, y in (("11. danni visibili", 1030), ("14. osservazioni", 1110)):
            d.text((x0 + 8, y - 44), label, fill=PRINT, font=f_sec)
            d.line((x0 + 8, y, x1 - 8, y), fill=RULE, width=1)

    # section 12: circumstances
    cx0, cx1 = CENTER
    d.line((cx0, 232, cx0, 1130), fill=PRINT, width=2)
    d.line((cx1, 232, cx1, 1130), fill=PRINT, width=2)
    d.text(((cx0 + cx1) // 2, 252), "12. circostanze dell'incidente", fill=PRINT, font=f_sec, anchor="mm")
    d.text(((cx0 + cx1) // 2, 276), "mettere una croce (X) soltanto nelle", fill=PRINT, font=f_small, anchor="mm")
    d.text(((cx0 + cx1) // 2, 292), "caselle utili alla descrizione", fill=PRINT, font=f_small, anchor="mm")
    f_circ = font(11)
    for n in range(1, 18):
        y = ROW0 + (n - 1) * ROW_H
        d.line((cx0, y, cx1, y), fill=(205, 210, 218), width=1)
        for s in SIDES:
            b = check_box(s, n)
            d.rectangle(b.as_tuple(), outline=PRINT, width=2)
        d.text((516, y + ROW_H // 2), str(n), fill=PRINT, font=f_small, anchor="mm")
        d.text((724, y + ROW_H // 2), str(n), fill=PRINT, font=f_small, anchor="mm")
        lines = _wrap(CIRCUMSTANCES[n - 1], f_circ, 178)[:3]
        top = y + ROW_H // 2 - (len(lines) * 12) // 2
        for i, line in enumerate(lines):
            d.text((620, top + i * 12), line, fill=PRINT, font=f_circ, anchor="ma")
    y = ROW0 + 17 * ROW_H
    d.line((cx0, y, cx1, y), fill=PRINT, width=2)
    d.text((620, y + 30), "indicare il numero di", fill=PRINT, font=f_small, anchor="mm")
    d.text((620, y + 46), "caselle barrate", fill=PRINT, font=f_small, anchor="mm")
    for s in SIDES:
        d.rectangle(TOTAL_BOX[s].as_tuple(), outline=PRINT, width=2)

    # section 13 sketch and section 15 signatures
    d.line((40, 1130, 1200, 1130), fill=PRINT, width=2)
    d.text((60, 1140), "13. grafico dell'incidente al momento dell'urto", fill=PRINT, font=f_sec)
    d.text((60, 1162), "indicare: il tracciato delle strade, la direzione di marcia di A e B, la loro posizione al "
           "momento dell'urto, i segnali stradali, i nomi delle strade", fill=RULE, font=f_small)
    sb = SKETCH_BOX
    for gx in range(sb.x0, sb.x1 + 1, 20):
        d.line((gx, sb.y0, gx, sb.y1), fill=(214, 226, 238), width=1)
    for gy in range(sb.y0, sb.y1 + 1, 20):
        d.line((sb.x0, gy, sb.x1, gy), fill=(214, 226, 238), width=1)
    d.rectangle(sb.as_tuple(), outline=RULE, width=1)
    d.line((40, 1530, 1200, 1530), fill=PRINT, width=2)
    d.text((60, 1540), "15. firma dei conducenti", fill=PRINT, font=f_sec)
    for s in SIDES:
        b = SIGN_BOX[s]
        d.rectangle(b.as_tuple(), outline=RULE, width=1)
        d.text((b.x0 - 12, (b.y0 + b.y1) // 2), s, fill=PRINT, font=font(22), anchor="rm")

    d.rectangle(FRAME, outline=PRINT, width=FRAME_WIDTH)
    d.text((620, 1730), "La denuncia non deve essere modificata dopo la firma e la separazione degli esemplari.",
           fill=RULE, font=f_small, anchor="mm")
    return img


def template_lines() -> Image.Image:
    """Mask (L) of the printed rules and boxes, used to score a registration."""
    g = blank().convert("L")
    return g.point(lambda v: 255 if v < 110 else 0)
