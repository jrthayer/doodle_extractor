"""Run with: python test_core.py

Builds a fake photo (dark desk, unevenly lit paper, gray grid, black
strokes, specks) and checks the pipeline and the GUI end to end.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

import doodle_core as core


def fake_photo():
    H, W = 900, 1200
    img = np.full((H, W), 45, np.float32)                  # dark desk
    paper = np.zeros((H, W), np.uint8)
    cv2.rectangle(paper, (100, 80), (1100, 820), 255, -1)
    xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    lit = 235 - 50 * xx                                     # lighting falls off to the right
    img = np.where(paper > 0, lit, img)

    grid = np.zeros((H, W), np.uint8)
    for x in range(120, 1100, 40):
        cv2.line(grid, (x, 80), (x, 820), 255, 1)
    for y in range(100, 820, 40):
        cv2.line(grid, (100, y), (1100, y), 255, 1)
    img = np.where(grid > 0, img * 0.82, img)               # faint gray grid

    ink = np.zeros((H, W), np.uint8)
    cv2.circle(ink, (400, 400), 150, 255, 6)
    cv2.line(ink, (650, 250), (950, 600), 255, 8)
    img = np.where(ink > 0, 30, img)

    specks = np.zeros((H, W), np.uint8)
    for (x, y) in [(300, 700), (800, 150), (1000, 750)]:
        cv2.circle(specks, (x, y), 2, 255, -1)
    img = np.where(specks > 0, 40, img)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), ink, grid, specks, paper


def test_pipeline():
    bgr, ink, grid, specks, paper = fake_photo()
    flat, mask = core.prepare(bgr)
    assert mask is not None, "page should be detected"

    rgba = core.process(flat, mask, core.Settings())
    assert rgba.shape == (*bgr.shape[:2], 4)
    assert (rgba[:, :, :3] == 255).all(), "color must be pure white"
    a = rgba[:, :, 3]

    inner_ink = cv2.erode(ink, np.ones((3, 3), np.uint8))
    assert a[inner_ink > 0].min() > 240, "strokes should be solid"
    near_ink = cv2.dilate(ink | specks, np.ones((9, 9), np.uint8))
    assert a[(grid > 0) & (near_ink == 0)].max() == 0, "grid should vanish"
    assert a[paper == 0].max() == 0, "desk should be transparent"
    assert a[specks > 0].max() > 0, "specks kept by default"

    clean = core.process(flat, mask, core.Settings(min_speck=50))[:, :, 3]
    assert clean[specks > 0].max() == 0, "specks removed"
    assert clean[inner_ink > 0].min() > 240, "strokes survive speck removal"

    cropped = core.process(flat, mask, core.Settings(min_speck=50, auto_crop=True, pad=8))
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(ink))
    ch, cw = cropped.shape[:2]
    assert abs(cw - (w + 16)) <= 4 and abs(ch - (h + 16)) <= 4, (cw, ch, w, h)
    print("pipeline ok")


def test_unicode_path():
    bgr = fake_photo()[0]
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "dessin_é_日本.png"
        cv2.imencode(".png", bgr)[1].tofile(str(src))
        loaded = core.load_image(src)
        assert np.array_equal(loaded, bgr)
        out = Path(d) / "sortie_é.png"
        core.save_png(out, core.to_rgba(np.full((4, 4), 7, np.uint8)))
        back = cv2.imdecode(np.fromfile(str(out), np.uint8), cv2.IMREAD_UNCHANGED)
        assert back.shape == (4, 4, 4) and back[0, 0, 3] == 7
    print("unicode paths ok")


def test_gui():
    import tkinter as tk
    from doodle_gui import DoodleApp

    bgr = fake_photo()[0]
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "photo.png"
        core.save_png(src, bgr)
        root = tk.Tk()
        root.withdraw()
        app = DoodleApp(root)
        root.update()
        assert app.open_path(src)
        app.white.set(150)
        app.min_speck.set(50)
        app.auto_crop.set(True)
        app.on_white()
        app.redraw()
        for mode in ("Dark", "Checker", "Light"):
            app.bg_mode.set(mode)
            app.redraw()
        out = Path(d) / "photo_ink.png"
        assert app.save_path(out)
        saved = cv2.imdecode(np.fromfile(str(out), np.uint8), cv2.IMREAD_UNCHANGED)
        assert saved.shape[2] == 4 and saved.shape[:2] != bgr.shape[:2], "cropped output"
        root.destroy()
    print("gui ok")


if __name__ == "__main__":
    test_pipeline()
    test_unicode_path()
    test_gui()
    print("all tests passed")
