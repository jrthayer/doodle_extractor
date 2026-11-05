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


def circle_loop(cx=400, cy=400, r=190, n=60):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1).astype(np.int32)


def test_pieces():
    bgr, ink, grid, specks, paper = fake_photo()
    flat, mask = core.prepare(bgr)
    s = core.Settings()

    # Loop around the circle only (the diagonal line starts at x=650)
    piece = core.process_piece(flat, mask, s, circle_loop())
    assert (piece[:, :, :3] == 255).all()
    ph, pw = piece.shape[:2]
    # circle: centre (400, 400), radius 150, 6px stroke -> ~306px across + 2 * 8 pad
    assert abs(pw - 322) <= 6 and abs(ph - 322) <= 6, (pw, ph)
    assert piece[:, :, 3].max() == 255, "circle ink present"

    # Loop around nothing: falls back to the loop's own bounds, fully transparent
    empty = core.process_piece(flat, mask, s, [(150, 600), (250, 600), (250, 680), (150, 680)])
    assert empty.shape[:2] == (81, 101) and empty[:, :, 3].max() == 0, empty.shape

    # Loop running off the image edge doesn't crash
    core.process_piece(flat, mask, s, [(-50, -50), (300, -50), (300, 300), (-50, 300)])
    print("pieces ok")


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
        root.geometry("1300x800")
        app = DoodleApp(root)
        root.update()
        assert app.open_path(src)
        root.update()
        app.white.set(150)
        app.min_speck.set(50)
        app.auto_crop.set(True)
        app.on_white()
        app.refresh()
        for mode in ("Dark", "Checker", "Light"):
            app.bg_mode.set(mode)
            app.refresh()

        # Zooming keeps the photo point under the mouse in place
        before = app.to_image(200, 150)
        app.zoom_at(200, 150, 2.0)
        after = app.to_image(200, 150)
        assert max(abs(before[0] - after[0]), abs(before[1] - after[1])) < 1e-6, (before, after)
        app.zoom_to(100)                       # clamped to the max
        assert app.scale == 8.0 and not app.fit
        app.zoom_to(0.0001)                    # clamped to fit
        assert app.fit
        app.zoom_to(1.0)
        app._pan_start(type("E", (), {"x": 300, "y": 300})())
        app._pan_move(type("E", (), {"x": 200, "y": 250})())
        app._pan_end(None)
        app.zoom_fit()
        out = Path(d) / "photo_ink.png"
        assert app.save_path(out)
        saved = cv2.imdecode(np.fromfile(str(out), np.uint8), cv2.IMREAD_UNCHANGED)
        assert saved.shape[2] == 4 and saved.shape[:2] != bgr.shape[:2], "cropped output"

        # Lasso two pieces with simulated mouse drags (image -> canvas coords)
        class Ev:
            def __init__(self, x, y):
                self.x, self.y = int(x), int(y)

        def drag(loop):
            pts = [app.to_canvas(x, y) for x, y in loop]
            app._lasso_start(Ev(*pts[0]))
            for p in pts[1:]:
                app._lasso_move(Ev(*p))
            app._lasso_end(Ev(*pts[-1]))

        app.min_speck.set(0)
        drag(circle_loop())
        drag([(620, 220), (980, 220), (980, 630), (620, 630)])   # around the line
        app._lasso_start(Ev(10, 10)); app._lasso_end(Ev(10, 10))  # a click: ignored
        assert len(app.pieces) == 2, len(app.pieces)
        root.update()

        p1 = Path(d) / "one.png"
        assert app.save_piece(0, p1)
        one = cv2.imdecode(np.fromfile(str(p1), np.uint8), cv2.IMREAD_UNCHANGED)
        assert abs(one.shape[1] - 322) <= 8, one.shape

        out_dir = Path(d) / "pieces"
        out_dir.mkdir()
        written = app.save_all(out_dir)
        assert [p.name for p in written] == ["photo_piece1.png", "photo_piece2.png"]
        line = cv2.imdecode(np.fromfile(str(written[1]), np.uint8), cv2.IMREAD_UNCHANGED)
        assert abs(line.shape[1] - (300 + 8 + 16)) <= 10, line.shape   # line is 300px wide

        # Zoomed in, a loop still lands in the right place on the photo
        app.zoom_to(2.0, 0, 0)
        app.vx, app.vy = 300, 300
        app._clamp_view()
        drag([(320, 320), (480, 320), (480, 480), (320, 480)])
        poly = app.pieces[-1]
        assert abs(poly[:, 0].min() - 320) <= 1 and abs(poly[:, 1].max() - 480) <= 1, poly

        app.select_piece(0)
        app.delete_piece(0)
        assert len(app.pieces) == 2 and app.selected == 0
        root.destroy()
    print("gui ok")


def test_preview_keeps_thin_lines():
    """At fit-to-window zoom on a big photo, thin lines must still show."""
    import tkinter as tk
    from doodle_gui import DoodleApp

    H, W = 3000, 4000
    img = np.full((H, W), 225, np.uint8)
    cv2.line(img, (500, 1500), (3500, 1500), 40, 2)        # 2px pen line
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "big.png"
        core.save_png(src, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        root = tk.Tk()
        root.geometry("1300x800")
        app = DoodleApp(root)
        root.update()
        app.open_path(src)
        root.update()
        app.zoom_fit()
        assert app.scale < 0.25, app.scale
        shown, _, _ = app._view_alpha()
        assert shown.max() > 60, f"line too faint in preview: {shown.max()}"
        root.destroy()
    print("thin lines ok")


if __name__ == "__main__":
    test_pipeline()
    test_pieces()
    test_unicode_path()
    test_gui()
    test_preview_keeps_thin_lines()
    print("all tests passed")
