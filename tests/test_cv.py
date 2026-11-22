"""Computer-vision agents on real synthetic pages: registration, crosses, signatures, the quality gate."""
import numpy as np
from conftest import case
from PIL import Image, ImageFilter

from cailens import form
from cailens.agents import checkbox, registrar, signature
from cailens.truth import AMBIGUOUS, expected


def test_scan_and_photo_register():
    for seed in (10000, 10003, 10012):
        _t, img = case(seed)
        reg = registrar.register(img)
        assert reg.ok and reg.score > 0.95, (seed, reg.score)
        assert reg.quality_ok


def test_marks_and_signatures_match_the_truth():
    for seed in (10001, 10003, 10010, 10017):
        t, img = case(seed)
        reg = registrar.register(img)
        exp = expected(t)
        for fid, r in checkbox.read(reg).items():
            if exp[fid] == AMBIGUOUS:
                assert r.value == "incerto", fid  # a scratched-out cross always goes to a person
            else:
                assert r.value == exp[fid], (seed, fid, r.evidence)
        for fid, r in signature.read(reg).items():
            assert r.value == exp[fid], (seed, fid, r.evidence)


def test_holdout_page_with_new_handwriting():
    t, img = case(90004, "holdout")
    reg = registrar.register(img)
    exp = expected(t)
    wrong = [f for f, r in checkbox.read(reg).items() if exp[f] != AMBIGUOUS and r.value not in (exp[f], "incerto")]
    assert wrong == []


def test_blank_or_noise_is_refused():
    blank = Image.new("RGB", (900, 1200), "white")
    assert not registrar.register(blank).ok
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (1200, 900, 3), dtype=np.uint8))
    assert not registrar.register(noise).ok


def test_blur_trips_the_quality_gate_and_caps_confidence():
    _t, img = case(10000)
    blurred = img.filter(ImageFilter.GaussianBlur(3.2))
    reg = registrar.register(blurred)
    assert not reg.quality_ok
    assert "sfocata" in reg.quality_note()
    assert all(r.confidence <= 0.5 for r in checkbox.read(reg).values())
    assert all(r.confidence <= 0.5 for r in signature.read(reg).values())


def test_fold_line_and_dots_are_not_crosses():
    reg = registrar.register(form.blank())
    # draw a crease across the circumstances and a pen dot inside an empty box
    y = form.check_box("A", 5).y0 + 12
    reg.ink[y:y + 2, :] = 1  # a 2 px crease: 32 px inside the box, wide - a stroke if lines were not removed
    b = form.check_box("B", 3)
    reg.ink[b.y0 + 10:b.y0 + 15, b.x0 + 10:b.x0 + 15] = 1  # 25 px, compact: a dot, not a cross
    reg.strokes = None
    assert checkbox.read(reg)["A.c05"].value == "no"
    assert checkbox.read(reg)["B.c03"].value == "no"


def test_classify_bands():
    assert checkbox.classify(0.0)[0] == "no"
    assert checkbox.classify(0.05)[0] == "incerto"
    assert checkbox.classify(0.3)[0] == "si"
    assert checkbox.classify(0.95)[0] == "incerto"
    assert signature.classify(10, 5)[0] == "no"
    assert signature.classify(150, 40)[0] == "incerto"
    assert signature.classify(900, 110)[0] == "si"
