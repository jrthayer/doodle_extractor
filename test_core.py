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


def test_doodles():
    bgr, ink, grid, specks, paper = fake_photo()
    flat, mask = core.prepare(bgr)
    s = core.Settings()

    # Loop around the circle only (the diagonal line starts at x=650)
    doodle = core.process_doodle(flat, mask, s, circle_loop())
    assert (doodle[:, :, :3] == 255).all()
    ph, pw = doodle.shape[:2]
    # circle: centre (400, 400), radius 150, 6px stroke -> ~306px across + 2 * 8 pad
    assert abs(pw - 322) <= 6 and abs(ph - 322) <= 6, (pw, ph)
    assert doodle[:, :, 3].max() == 255, "circle ink present"

    # Loop around nothing: falls back to the loop's own bounds, fully transparent
    empty = core.process_doodle(flat, mask, s, [(150, 600), (250, 600), (250, 680), (150, 680)])
    assert empty.shape[:2] == (81, 101) and empty[:, :, 3].max() == 0, empty.shape

    # Loop running off the image edge doesn't crash
    core.process_doodle(flat, mask, s, [(-50, -50), (300, -50), (300, 300), (-50, 300)])

    # Line color and background: white on transparent by default
    a = np.array([[0, 128, 255]], np.uint8)
    white = core.to_rgba(a)
    assert (white[:, :, :3] == 255).all() and (white[:, :, 3] == a).all()
    red = core.to_rgba(a, (255, 0, 0))
    assert red[0, 0].tolist()[:3] == [0, 0, 255], "BGR order"
    assert core.apply_background(red, None) is red, "transparent: unchanged"
    on_black = core.apply_background(red, (0, 0, 0))
    assert on_black[:, :, 3].tolist() == [[255, 255, 255]]
    assert on_black[0, 0, 2] == 0 and on_black[0, 2, 2] == 255 and 120 <= on_black[0, 1, 2] <= 130
    print("doodles ok")


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
        assert app.white.get() == 255, "the app starts with White at 255"
        app.white.set(160)             # these tests were written for White 160
        app.set_extract(True)          # these tests check extracted ink
        app.set_godot(True)            # ...as white ink on transparent
        app.detect_paper.set(True)     # ...and were written with paper detection on
        app.reprepare()
        root.update()
        app.white.set(150)
        app.min_speck.set(50)
        app.on_white()
        app.refresh()
        for mode in ("Black", "White", "Custom…", "Transparent"):   # preview with each
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
        assert saved.shape == (*bgr.shape[:2], 4), "no crop: whole image"
        # ...and the same through the pipeline the CLI uses
        flat, mask = core.prepare(bgr)
        assert np.array_equal(saved, core.process(flat, mask, app.settings()))

        # Lasso two doodles with simulated mouse drags (image -> canvas coords)
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
        assert len(app.doodles) == 2, len(app.doodles)
        root.update()

        p1 = Path(d) / "one.png"
        assert app.save_doodle(0, p1)
        one = cv2.imdecode(np.fromfile(str(p1), np.uint8), cv2.IMREAD_UNCHANGED)
        assert abs(one.shape[1] - 306) <= 8, one.shape          # circle, no margin

        out_dir = Path(d) / "doodles"
        out_dir.mkdir()
        written = app.save_all(out_dir)
        assert [p.name for p in written] == ["photo_doodle1.png", "photo_doodle2.png"]
        line = cv2.imdecode(np.fromfile(str(written[1]), np.uint8), cv2.IMREAD_UNCHANGED)
        assert abs(line.shape[1] - (300 + 8)) <= 10, line.shape   # line: 300px wide, 8px thick

        # Zoomed in, a loop still lands in the right place on the photo
        app.zoom_to(2.0, 0, 0)
        app.vx, app.vy = 300, 300
        app._clamp_view()
        drag([(320, 320), (480, 320), (480, 480), (320, 480)])
        poly = app.doodle_poly(-1)
        assert abs(poly[:, 0].min() - 320) <= 1 and abs(poly[:, 1].max() - 480) <= 1, poly

        app.select_doodle(0)
        app.delete_doodle(0)
        assert len(app.doodles) == 2 and app.selected == 0
        root.destroy()
    print("gui ok")


def test_rotation_core():
    bgr = fake_photo()[0]
    flat, mask = core.prepare(bgr)
    s = core.Settings(min_speck=50)
    base = core.process(flat, mask, s)

    # Quarter turns are lossless: same as rotating the finished PNG
    for q, code in [(1, cv2.ROTATE_90_CLOCKWISE), (2, cv2.ROTATE_180),
                    (3, cv2.ROTATE_90_COUNTERCLOCKWISE)]:
        rf, rm = core.rotate_prepared(flat, mask, q)
        assert np.array_equal(core.process(rf, rm, s), cv2.rotate(base, code)), q

    # Any angle: canvas grows, corners are transparent, ink survives
    rf, rm = core.rotate_prepared(flat, mask, 0, 12.5)
    out = core.process(rf, rm, s)
    assert out.shape[0] > base.shape[0] and out.shape[1] > base.shape[1]
    a = out[:, :, 3]
    assert a[:20, :20].max() == 0 and a[-20:, -20:].max() == 0, "corners transparent"
    assert (a > 240).sum() > 0.8 * (base[:, :, 3] > 240).sum(), "ink kept"

    print("rotation core ok")


def test_doodle_source():
    """A doodle loaded on its own converts the same as cutting it from the page."""
    bgr = fake_photo()[0]
    flat, mask = core.prepare(bgr)
    s = core.Settings(auto_crop=True)
    loop = circle_loop()                          # circle only, not the line

    sub, sub_mask = core.doodle_source(flat, mask, loop)
    assert sub.shape == (2 * 190 + 2 * 24 + 1, 2 * 190 + 2 * 24 + 1), sub.shape
    alone = core.process(sub, sub_mask, s)
    cut = core.process_doodle(flat, mask, s, loop)
    assert abs(alone.shape[0] - cut.shape[0]) <= 2 and abs(alone.shape[1] - cut.shape[1]) <= 2
    assert sub[0, 0] == 255 and sub[-1, -1] == 255, "outside the loop is paper"
    print("doodle source ok")


def test_rotation_gui():
    import tkinter as tk
    from doodle_gui import DoodleApp

    bgr = fake_photo()[0]      # 1200 x 900, circle centred at (400, 400)
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "photo.png"
        core.save_png(src, bgr)
        root = tk.Tk()
        root.geometry("1300x800")
        app = DoodleApp(root)
        root.update()
        app.open_path(src)
        app.white.set(160)             # these tests were written for White 160
        app.set_extract(True)          # these tests check extracted ink
        app.set_godot(True)            # ...as white ink on transparent
        app.detect_paper.set(True)     # ...and were written with paper detection on
        app.reprepare()
        root.update()

        app.add_doodle(circle_loop())

        def centre():
            p = app.doodle_poly(0)
            return p[:, 0].mean(), p[:, 1].mean()

        # Rotate the photo 90 degrees clockwise: the loop follows the circle
        app.rotate_photo(1)
        assert app.flat.shape == (1200, 900)
        cx, cy = centre()
        assert abs(cx - (899 - 400)) < 2 and abs(cy - 400) < 2, (cx, cy)
        out = Path(d) / "circle.png"
        assert app.save_doodle(0, out)
        img = cv2.imdecode(np.fromfile(str(out), np.uint8), cv2.IMREAD_UNCHANGED)
        assert abs(img.shape[0] - 306) <= 6 and abs(img.shape[1] - 306) <= 6, img.shape

        # Straighten slider: still the same circle
        app.angle.set(20)
        app.apply_rotation()
        assert app.save_doodle(0, out)
        img = cv2.imdecode(np.fromfile(str(out), np.uint8), cv2.IMREAD_UNCHANGED)
        assert abs(img.shape[0] - 306) <= 8 and abs(img.shape[1] - 306) <= 8, img.shape

        app.reset_rotation()
        cx, cy = centre()
        assert abs(cx - 400) < 2 and abs(cy - 400) < 2, (cx, cy)

        root.update()
        root.destroy()
    print("rotation gui ok")


def test_edit_doodle_window():
    import tkinter as tk
    import doodle_gui
    from doodle_gui import DoodleApp

    def read(p):
        return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)

    bgr = fake_photo()[0]
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "photo.png"
        core.save_png(src, bgr)
        root = tk.Tk()
        root.geometry("1300x800")
        app = DoodleApp(root)
        root.update()
        app.open_path(src)
        app.white.set(160)             # these tests were written for White 160
        app.set_extract(True)          # these tests check extracted ink
        app.set_godot(True)            # ...as white ink on transparent
        app.detect_paper.set(True)     # ...and were written with paper detection on
        app.reprepare()
        app.white.set(150)
        app.refresh()
        app.add_doodle(circle_loop())
        app.add_doodle([(620, 220), (980, 220), (980, 630), (620, 630)])   # the line
        root.update()

        main_out = Path(d) / "main.png"
        assert app.save_doodle(1, main_out)

        editor = app.edit_doodle(1)
        root.update()
        assert editor.is_doodle and isinstance(editor.root, tk.Toplevel)
        assert "doodle 2" in editor.root.title()
        assert editor.white.get() == 150, "settings carried over"
        assert editor.crop_fit_saved, "editor starts cropped to the drawing"
        assert "disabled" in editor.paper_check.state()
        assert editor in doodle_gui._APPS and app in doodle_gui._APPS

        # Saving from the editor gives the same doodle as saving it from the page
        ed_out = Path(d) / "edited.png"
        assert editor.save_path(ed_out)
        a, b = read(main_out), read(ed_out)
        assert abs(a.shape[0] - b.shape[0]) <= 2 and abs(a.shape[1] - b.shape[1]) <= 2, (a.shape, b.shape)

        # The editor has its own settings and can rotate the doodle
        editor.white.set(120)
        assert app.white.get() == 150
        editor.rotate_photo(1)
        assert editor.save_path(ed_out)
        c = read(ed_out)
        assert abs(c.shape[0] - b.shape[1]) <= 2 and abs(c.shape[1] - b.shape[0]) <= 2, (b.shape, c.shape)
        assert editor._doodle_name(0) == "photo_doodle2_doodle1.png"

        # Opening a photo in the editor turns it back into a normal window
        editor.open_path(src)
        assert not editor.is_doodle and "disabled" not in editor.paper_check.state()

        editor.root.destroy()
        root.update()
        assert editor not in doodle_gui._APPS and app in doodle_gui._APPS
        root.destroy()
        assert app not in doodle_gui._APPS
    print("edit doodle window ok")


def test_crop_tool():
    import tkinter as tk
    import doodle_gui
    from doodle_gui import DoodleApp

    def read(p):
        return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)

    class Ev:
        def __init__(self, x, y):
            self.x, self.y = int(round(x)), int(round(y))

    def near(a, b, tol=3):
        return all(abs(p - q) <= tol for p, q in zip(a, b))

    bgr = fake_photo()[0]                     # 1200 x 900, circle (400, 400) r=150
    W, H = 1200, 900
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "photo.png"
        core.save_png(src, bgr)
        root = tk.Tk()
        root.geometry("1300x800")
        app = DoodleApp(root)
        root.update()
        app.open_path(src)
        app.white.set(160)             # these tests were written for White 160
        app.set_extract(True)          # these tests check extracted ink
        app.set_godot(True)            # ...as white ink on transparent
        app.detect_paper.set(True)     # ...and were written with paper detection on
        app.reprepare()
        root.update()
        app.add_doodle(circle_loop())          # made before cropping

        def drag(p0, p1):
            a, b = app.to_canvas(*p0), app.to_canvas(*p1)
            app._crop_press(Ev(*a))
            app._crop_move(Ev(*b))
            app._crop_release(Ev(*b))

        # Hidden until Crop is clicked; box starts at the image edges
        assert not app.cropping and not app.crop_section.winfo_manager()
        app.start_crop()
        root.update()
        assert app.cropping and app.crop_section.winfo_manager()
        assert not app.tips.winfo_manager() and "pressed" in app.crop_button.state()
        assert app.edit_rect() == (0, 0, W, H)
        assert [app.crop_vars[k].get() for k in "xywh"] == ["0", "0", "1200", "900"]

        # Typed values are clamped to the image
        app.crop_vars["x"].set("100")
        app.crop_vars["w"].set("5000")
        app._crop_from_spins()
        assert app.edit_rect() == (100, 0, 1100, 900)

        # Drag the right edge, then the top-left corner, then move the box
        drag((1200, 450), (1000, 450))
        assert near(app.edit_rect(), (100, 0, 900, 900)), app.edit_rect()
        drag((100, 0), (150, 50))
        assert near(app.edit_rect(), (150, 50, 850, 850)), app.edit_rect()
        x, y, w, h = app.edit_rect()
        drag((x + w / 2, y + h / 2), (x + w / 2 - 30, y + h / 2 - 20))
        assert near(app.edit_rect(), (x - 30, y - 20, w, h)), app.edit_rect()
        assert app.crop_vars["x"].get() == str(app.edit_rect()[0]), "spinboxes follow drags"
        drag((x + w / 2, y + h / 2), (x + w / 2, y + h / 2 + 500))   # can't leave the image
        assert app.edit_rect()[1] + app.edit_rect()[3] == H
        assert app._crop_hit(*app.to_canvas(1150, 20)) is None, "outside the box"

        # "Crop to paper" snaps to the ink box; switching it off restores the box
        before = app.edit_rect()
        app.crop_fit.set(True)
        app.on_crop_fit()
        assert app.edit_rect() == core.crop_box(app._drawing_alpha(), app.pad.get())
        app.pad.set(30)
        app.refresh()
        assert app.edit_rect() == core.crop_box(app._drawing_alpha(), 30), "box tracks the ink"
        app.pad.set(0)
        app.crop_fit.set(False)
        app.on_crop_fit()
        assert app.edit_rect() == before
        app.crop_fit.set(True)
        app.on_crop_fit()
        app.set_crop_box(230, 220, 340, 360)  # hand edit turns "to the drawing" off
        assert not app.crop_fit.get()

        # Save crop: the working image is the cropped area
        app.save_crop()
        root.update()
        assert not app.cropping and not app.crop_section.winfo_manager()
        assert app.tips.winfo_manager() and "pressed" not in app.crop_button.state()
        assert app.flat.shape == (360, 340)
        out = Path(d) / "cropped.png"
        assert app.save_path(out)
        saved = read(out)
        assert saved.shape == (360, 340, 4)
        assert np.array_equal(saved[:, :, 3], app.alpha_full[220:580, 230:570])

        # A doodle made before the crop still points at the same drawing
        p = app.doodle_poly(0)
        assert abs(p[:, 0].mean() - (400 - 230)) < 2 and abs(p[:, 1].mean() - (400 - 220)) < 2
        assert app.save_doodle(0, out)
        assert abs(read(out).shape[0] - 306) <= 6, read(out).shape

        # Reopening shows the saved box; Reset puts it back at the edges
        app.crop_button.invoke()
        assert app.cropping and app.edit_rect() == (230, 220, 340, 360)
        assert app.flat.shape == (H, W)
        app.crop_fit.set(True)
        app.on_crop_fit()
        app.reset_crop()
        assert app.edit_rect() == (0, 0, W, H) and not app.crop_fit.get()
        # Clicking Crop again collapses the section and keeps the changes
        app.set_crop_box(100, 100, 500, 400)
        app.crop_button.invoke()
        root.update()
        assert not app.cropping and not app.crop_section.winfo_manager()
        assert app.flat.shape == (400, 500)
        # Esc does the same
        app.start_crop()
        app.set_crop_box(230, 220, 340, 360)
        app._on_escape()
        assert not app.cropping and app.flat.shape == (360, 340)

        # Saving is refused while cropping
        shown = []
        orig_info = doodle_gui.messagebox.showinfo
        doodle_gui.messagebox.showinfo = lambda *a, **k: shown.append(a)
        try:
            app.start_crop()
            assert not app.save_path(out) and shown
            app._on_return()                  # Enter saves the crop
            assert not app.cropping
        finally:
            doodle_gui.messagebox.showinfo = orig_info

        # Rotation: the crop follows the content, exactly for 90-degree turns
        app.rotate_photo(1)
        assert app.flat.shape == (340, 360)
        assert app.save_path(out)
        assert np.array_equal(read(out), cv2.rotate(saved, cv2.ROTATE_90_CLOCKWISE))
        app.rotate_photo(-1)
        assert app.work_rect == (230, 220, 340, 360)
        app.angle.set(7)
        app.apply_rotation()
        app.angle.set(0)
        app.apply_rotation()
        assert app.work_rect == (230, 220, 340, 360), "straighten there and back: no growth"

        # A saved "to the drawing" crop stays live
        app.start_crop()
        app.crop_fit.set(True)
        app.on_crop_fit()
        app.save_crop()
        assert app.crop_fit_saved and app.work_rect == core.crop_box(app._drawing_alpha(), 0)
        app.pad.set(40)
        app.refresh()
        assert app.work_rect == core.crop_box(app._drawing_alpha(), 40)

        # Saving the full box means no crop
        app.start_crop()
        app.crop_fit.set(False)
        app.on_crop_fit()                     # restores the remembered box: whole image
        app.save_crop()
        assert app.crop_corners is None and not app.crop_fit_saved
        assert app.flat.shape == (H, W)
        root.destroy()
    print("crop tool ok")


def test_crop_to_paper():
    """With the app's defaults (Detect paper off, White 255, no detection),
    "Crop to paper" still snaps tight to the ink, not the whole photo."""
    import tkinter as tk
    from doodle_gui import DoodleApp

    bgr, ink = fake_photo()[:2]
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(ink))    # the drawing's real box
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "photo.png"
        core.save_png(src, bgr)
        root = tk.Tk()
        root.geometry("1300x800")
        app = DoodleApp(root)
        root.update()
        app.open_path(src)
        assert not app.detect_paper.get() and app.white.get() == 255 and not app.extract.get()

        def fitted():
            app.start_crop()
            app.crop_fit.set(True)
            app.on_crop_fit()
            box = app.edit_rect()
            app.crop_fit.set(False)
            app.on_crop_fit()
            app.save_crop()
            return box

        for setup in ("defaults", "paper on", "detection on"):
            if setup == "paper on":
                app.detect_paper.set(True)
                app.reprepare()
            elif setup == "detection on":
                app.set_extract(True)
            bx, by, bw, bh = fitted()
            assert (abs(bx - x) <= 4 and abs(by - y) <= 4 and abs(bw - w) <= 8
                    and abs(bh - h) <= 8), (setup, (bx, by, bw, bh), (x, y, w, h))
        root.destroy()
    print("crop to paper ok")


def test_mouse_modes():
    import tkinter as tk
    from doodle_gui import DoodleApp

    class Ev:
        def __init__(self, x, y):
            self.x, self.y = int(round(x)), int(round(y))

    bgr = fake_photo()[0]
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "photo.png"
        core.save_png(src, bgr)
        root = tk.Tk()
        root.geometry("1300x800")
        app = DoodleApp(root)
        root.update()
        app.open_path(src)
        app.white.set(160)             # these tests were written for White 160
        app.set_extract(True)          # these tests check extracted ink
        app.set_godot(True)            # ...as white ink on transparent
        app.detect_paper.set(True)     # ...and were written with paper detection on
        app.reprepare()
        root.update()

        def left_drag(points):
            pts = [app.to_canvas(x, y) for x, y in points]
            app._left_press(Ev(*pts[0]))
            for p in pts[1:]:
                app._left_move(Ev(*p))
            app._left_release(Ev(*pts[-1]))

        # Default: left-drag moves the image, no doodles
        assert not app.select_mode.get() and str(app.canvas.cget("cursor")) == "fleur"
        app.zoom_to(1.0)
        vx, vy = app.vx, app.vy
        app._left_press(Ev(400, 300))
        app._left_move(Ev(300, 250))
        app._left_release(Ev(300, 250))
        assert app.vx > vx and app.vy > vy and not app.doodles, (vx, vy, app.vx, app.vy)
        assert str(app.canvas.cget("cursor")) == "fleur"
        app.zoom_fit()

        # Lasso doodle: loops make doodles, and the mode stays on
        app.select_button.invoke()
        root.update()
        assert str(app.canvas.cget("cursor")) == "crosshair"
        left_drag(circle_loop())
        left_drag([(620, 220), (980, 220), (980, 630), (620, 630)])
        assert len(app.doodles) == 2 and app.select_mode.get()

        # Esc cancels a loop in progress first, then leaves select mode
        p0 = app.to_canvas(100, 100)
        app._left_press(Ev(*p0))
        app._left_move(Ev(p0[0] + 50, p0[1]))
        app._on_escape()
        assert app._lasso is None and app.select_mode.get()
        app._left_release(Ev(p0[0] + 50, p0[1] + 50))
        assert len(app.doodles) == 2
        app._on_escape()
        assert not app.select_mode.get() and str(app.canvas.cget("cursor")) == "fleur"

        # Crop mode turns select mode off and owns the left button
        app.set_select_mode(True)
        app.start_crop()
        assert not app.select_mode.get()
        app.set_select_mode(True)                   # can't switch on while cropping
        assert not app.select_mode.get()
        a = app.to_canvas(1200, 450)
        b = app.to_canvas(1000, 450)
        app._left_press(Ev(*a))
        app._left_move(Ev(*b))
        app._left_release(Ev(*b))
        assert abs(app.edit_rect()[2] - 1000) <= 3 and len(app.doodles) == 2
        app.reset_crop()
        app.toggle_crop()                           # full box: no crop
        assert app.crop_corners is None and str(app.canvas.cget("cursor")) == "fleur"
        root.destroy()
    print("mouse modes ok")


def test_extract_toggle():
    import tkinter as tk
    from doodle_gui import DoodleApp

    def read(p):
        return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)

    # A colored photo, so "the photo as is" is easy to tell apart from ink
    bgr = fake_photo()[0].copy()
    bgr[:, :, 0] = (bgr[:, :, 0] * 0.6).astype(np.uint8)       # tint it
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "photo.png"
        core.save_png(src, bgr)
        root = tk.Tk()
        root.geometry("1300x800")
        app = DoodleApp(root)
        root.update()
        app.open_path(src)
        root.update()
        out = Path(d) / "out.png"

        # Default: no detection. Section hidden, button up, photo saved as is
        assert not app.extract.get() and not app.extract_section.winfo_manager()
        assert "pressed" not in app.extract_button.state()
        assert app.extract_button.cget("text") == "Detect lines"
        assert app.apply_check.cget("text") == "Apply detection"
        assert "ignore background" in app.paper_check.cget("text")
        assert not app.tint_box.winfo_manager(), "line color only shows while detecting"
        assert app.bg_mode.get() == "White" and app.tint == (0, 0, 0), "black on white"
        app.bg_mode.set("Transparent")      # the photo checks below look at its alpha
        app.on_background()
        assert app.save_path(out)
        saved = read(out)
        assert saved.shape == (900, 1200, 4) and (saved[:, :, 3] == 255).all()
        assert np.array_equal(saved[:, :, :3], bgr), "unfiltered photo"

        # Raw doodle: the photo inside the loop, transparent outside
        app.add_doodle(circle_loop())
        assert app.save_doodle(0, out)
        doodle = read(out)
        assert doodle.shape[:2] == (381, 381), doodle.shape      # loop bounds (r=190)
        assert doodle[190, 190, 3] == 255 and doodle[0, 0, 3] == 0
        assert np.array_equal(doodle[190, 190, :3], bgr[400, 400])

        # The button only opens the settings: just "Apply detection" until it's on
        app.white.set(150)
        app.extract_button.invoke()
        root.update()
        assert app.extract_section.winfo_manager() and "pressed" in app.extract_button.state()
        assert not app.extract.get() and not app.extract_body.winfo_manager()
        assert app.save_path(out) and np.array_equal(read(out)[:, :, :3], bgr), "still the photo"

        # Apply detection: the rest of the settings show, the button gets a ✓, and
        # the lines are saved in the line color (black)
        app.apply_check.invoke()
        root.update()
        assert app.extract.get() and app.extract_body.winfo_manager()
        assert app.extract_button.cget("text") == "Detect lines ✓"
        assert app.tint_box.winfo_manager() and app.godot_button.winfo_manager()
        flat, mask = core.prepare(bgr, app.detect_paper.get())
        want = core.process(flat, mask, app.settings())
        assert app.save_path(out)
        ink = read(out)
        assert (ink[:, :, :3] == 0).all() and np.array_equal(ink[:, :, 3], want[:, :, 3])

        # Apply Godot: white lines on transparent, colors locked; off restores them
        app.bg_mode.set("Black")
        app.on_background()
        app.godot_button.invoke()
        assert app.godot and "pressed" in app.godot_button.state()
        assert app.bg_mode.get() == "Transparent" and app.tint == (255, 255, 255)
        assert "disabled" in app.bg_combo.state()
        assert str(app.tint_swatch.cget("state")) == "disabled"
        assert app.save_path(out) and np.array_equal(read(out), want), "Godot-ready ink"
        app.godot_button.invoke()
        assert not app.godot and app.bg_mode.get() == "Black" and app.tint == (0, 0, 0)
        assert "disabled" not in app.bg_combo.state()
        app.bg_mode.set("Transparent")
        app.on_background()
        app.set_godot(True)

        # Closing the settings keeps extraction on
        app.extract_button.invoke()
        root.update()
        assert not app.extract_section.winfo_manager() and app.extract_holder.winfo_height() <= 1
        assert app.extract.get() and app.extract_button.cget("text") == "Detect lines ✓"
        assert app.save_path(out) and (read(out)[:, :, :3] == 255).all()

        # Apply detection off: settings fold away, photo is back, slider values kept,
        # and Apply Godot goes off with it (colors restored)
        app.extract_button.invoke()
        app.apply_check.invoke()
        root.update()
        assert not app.extract.get() and not app.extract_body.winfo_manager()
        assert app.extract_button.cget("text") == "Detect lines"
        assert not app.godot and app.tint == (0, 0, 0) and app.bg_mode.get() == "Transparent"
        app.extract_button.invoke()
        assert app.white.get() == 150, "slider value remembered"
        assert app.save_path(out) and np.array_equal(read(out)[:, :, :3], bgr)

        # Rotation applies to the photo too; corners exposed by straightening are clear
        app.rotate_photo(1)
        assert app.save_path(out)
        assert np.array_equal(read(out)[:, :, :3], cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE))
        app.angle.set(10)
        app.apply_rotation()
        assert app.save_path(out)
        tilted = read(out)
        assert tilted[0, 0, 3] == 0 and tilted[tilted.shape[0] // 2, tilted.shape[1] // 2, 3] == 255
        app.reset_rotation()

        # Crop applies to the photo
        app.start_crop()
        app.set_crop_box(100, 50, 300, 200)
        app.save_crop()
        assert app.save_path(out)
        assert np.array_equal(read(out)[:, :, :3], bgr[50:250, 100:400])
        app.start_crop()
        app.set_crop_box(0, 0, 1200, 900)
        app.save_crop()

        # Editors inherit the state; an unfiltered editor shows the photo doodle
        editor = app.edit_doodle(0)
        root.update()
        assert not editor.extract.get() and editor.white.get() == 150
        assert editor.save_path(out)
        ed = read(out)
        assert ed[:, :, 3].max() == 255 and ed[0, 0, 3] == 0, "transparent outside the loop"
        editor.set_extract(True)
        assert editor.save_path(out) and (read(out)[:, :, :3] == 0).all(), "black lines"
        editor.root.destroy()
        app.set_extract(True)
        app.set_godot(True)
        editor = app.edit_doodle(0)
        assert editor.extract.get() and editor.extract_button.cget("text") == "Detect lines ✓"
        assert not editor.extract_section.winfo_manager(), "its settings start closed"
        assert editor.godot and editor.tint == (255, 255, 255)
        editor.set_godot(False)                 # ...and restores the main window's colors
        assert editor.tint == (0, 0, 0) and editor.bg_mode.get() == "Transparent"
        editor.root.destroy()
        root.destroy()
    print("extract toggle ok")


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
        app.white.set(160)             # these tests were written for White 160
        app.set_extract(True)          # these tests check extracted ink
        app.set_godot(True)            # ...as white ink on transparent
        app.detect_paper.set(True)     # ...and were written with paper detection on
        app.reprepare()
        root.update()
        app.zoom_fit()
        assert app.scale < 0.25, app.scale
        shown, _, _ = app._view_alpha()
        assert shown.max() > 60, f"line too faint in preview: {shown.max()}"
        root.destroy()
    print("thin lines ok")


def test_theme():
    import tkinter as tk
    from tkinter import ttk
    import doodle_gui
    from doodle_gui import DoodleApp, set_theme, DARK, LIGHT

    bgr = fake_photo()[0]
    start = doodle_gui._THEME["dark"]
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "photo.png"
        core.save_png(src, bgr)
        root = tk.Tk()
        root.geometry("1300x800")
        app = DoodleApp(root)
        root.update()
        app.open_path(src)
        app.doodles.append(doodle_gui.Doodle(np.array(circle_loop(), float)))
        app._rebuild_rows()
        editor = app.edit_doodle(0)
        root.update()

        set_theme(True)          # every open window switches
        root.update()
        assert ttk.Style(root).theme_use() == "clam"
        assert "light" in app.theme_tip and app.theme_button.cget("image")
        assert (ttk.Style(root).lookup("TButton", "background", ["pressed"])
                == DARK["pressed"])
        assert str(app.smooth_scale.cget("bg")) == DARK["pressed"], "settings highlight"
        switch = lambda: str(ttk.Style(root).layout("Opts.TCheckbutton"))
        assert "Switch.dark.indicator" in switch(), "on/off settings are sliding switches"
        hug0 = app.hug.get()
        app.hug_check.invoke()
        assert app.hug.get() != hug0
        app.hug_check.invoke()
        for a in (app, editor):
            assert a.dark.get() and a.canvas.cget("bg") == DARK["canvas"]
            assert str(a._opt_scales[0].cget("bg")) == DARK["pressed"]
            assert str(a.tips.cget("foreground")) == DARK["muted"]
        assert app._rows[0][0].cget("bg") == DARK["row"]
        app.select_doodle(0)
        assert app._rows[0][0].cget("bg") == DARK["row_sel"]
        assert app._rows[0][2].cget("fg") == DARK["fg"]
        app._rebuild_rows()       # new rows use the theme too
        assert app._rows[0][0].cget("bg") == DARK["row_sel"]

        # A window opened later starts dark
        editor2 = app.edit_doodle(0)
        assert editor2.dark.get() and editor2.canvas.cget("bg") == DARK["canvas"]

        set_theme(False)
        root.update()
        assert ttk.Style(root).theme_use() == "clam"     # recolorable, like dark mode
        assert "Switch.light.indicator" in switch()
        assert "dark" in app.theme_tip
        assert (ttk.Style(root).lookup("TButton", "background", ["pressed"])
                == LIGHT["pressed"])
        app.theme_button.event_generate("<Button-1>")   # clicking the sun/moon toggles
        assert doodle_gui._THEME["dark"] and editor.dark.get()
        app.toggle_theme()
        assert not doodle_gui._THEME["dark"]
        assert not app.dark.get() and app.canvas.cget("bg") == LIGHT["canvas"]
        assert app._rows[0][0].cget("bg") == LIGHT["row_sel"]
        for pal in (LIGHT, DARK):   # selected card and "on" state are gray shades, not colors
            for key in ("row_sel", "pressed"):
                r, g, b = (int(pal[key][k:k + 2], 16) for k in (1, 3, 5))
                assert r == g == b, (key, pal[key])
        root.destroy()
    doodle_gui._THEME["dark"] = start
    print("theme ok")


def test_hug_loop():
    import time
    import tkinter as tk
    from doodle_gui import DoodleApp

    alpha = np.zeros((900, 1200), np.uint8)
    cv2.circle(alpha, (400, 400), 150, 255, 6)           # a ring drawing
    loop = np.array(circle_loop(), np.int32)
    t = time.perf_counter()
    hug = core.hug_loop(alpha, loop)
    ms = (time.perf_counter() - t) * 1000
    assert hug is not None and hug.dtype == np.int32
    area = lambda p: cv2.contourArea(p.reshape(-1, 1, 2).astype(np.float32))
    assert area(hug) < area(loop) * 0.8, (area(hug), area(loop))
    # Keeps all the ink, the inside of the ring, and stays inside the loop
    full = core.cut_doodle(alpha, loop, 0).astype(int).sum()
    assert abs(core.cut_doodle(alpha, hug, 0).astype(int).sum() - full) <= full * 0.01
    lp = loop.reshape(-1, 1, 2).astype(np.float32)
    assert all(cv2.pointPolygonTest(lp, (float(x), float(y)), True) >= -2 for x, y in hug)
    assert cv2.pointPolygonTest(hug.reshape(-1, 1, 2), (400.0, 400.0), False) > 0, "holes filled"
    # It hugs: every outline point is close to the ring
    d = np.hypot(hug[:, 0] - 400, hug[:, 1] - 400)
    assert d.max() < 165 and d.min() > 140, (d.min(), d.max())

    # Two separate parts end up in one outline
    two = np.zeros((900, 1200), np.uint8)
    cv2.circle(two, (300, 400), 40, 255, 5)
    cv2.rectangle(two, (600, 380), (680, 440), 255, 5)
    box = np.array([(200, 300), (760, 300), (760, 520), (200, 520)], np.int32)
    hug2 = core.hug_loop(two, box)
    poly = hug2.reshape(-1, 1, 2)
    for pt in ((300.0, 400.0), (640.0, 410.0)):
        assert cv2.pointPolygonTest(poly, pt, False) > 0, pt
    assert area(hug2) < area(box) * 0.7

    assert core.hug_loop(np.zeros((900, 1200), np.uint8), loop) is None, "no ink: keep loop"

    # Solid black bars on red: hug the looped bar, leave out the neighbour it clips
    red = np.zeros((600, 800, 3), np.uint8)
    red[:] = (40, 40, 200)
    bar1, bar2 = np.zeros((600, 800), np.uint8), np.zeros((600, 800), np.uint8)
    cv2.rectangle(bar1, (200, 150), (330, 420), 255, -1)
    cv2.rectangle(bar2, (390, 150), (520, 420), 255, -1)
    red[(bar1 | bar2) > 0] = (30, 30, 30)
    faint = np.zeros((600, 800), np.uint8)              # what the ink detection sees: edges only
    faint[cv2.morphologyEx(bar1 | bar2, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0] = 120
    t_ = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    bar_loop = np.stack([265 + 170 * np.cos(t_), 285 + 190 * np.sin(t_)], 1).astype(np.int32)
    h = core.hug_loop(faint, bar_loop, color=red)
    assert core.cut_doodle(bar2, h, 0).max() == 0, "the clipped neighbour is left out"
    inner = cv2.erode(bar1, np.ones((5, 5), np.uint8))
    assert core.cut_doodle(inner, h, 0).astype(int).sum() >= inner.astype(int).sum() * 0.99
    assert area(h) < cv2.countNonZero(bar1) * 1.6, (area(h), cv2.countNonZero(bar1))

    # A solid W: its valleys stay open (traced, not bridged), at moderate smoothing too
    wpic = np.zeros((600, 800, 3), np.uint8)
    wpic[:] = (40, 40, 200)
    wmask = np.zeros((600, 800), np.uint8)
    cv2.polylines(wmask, [np.array([(200, 150), (290, 450), (400, 220), (510, 450), (600, 150)])],
                  False, 255, 60)
    wpic[wmask > 0] = (30, 30, 30)
    w_loop = np.stack([400 + 300 * np.cos(t_), 300 + 230 * np.sin(t_)], 1).astype(np.int32)
    for sm in (0, 30):
        h = core.hug_loop(np.zeros((600, 800), np.uint8), w_loop, smooth=sm, color=wpic)
        poly = h.reshape(-1, 1, 2)
        for valley in ((290, 250), (510, 250), (400, 420)):   # red, between the arms
            assert cv2.pointPolygonTest(poly, valley, False) < 0, (sm, valley)
        inner = cv2.erode(wmask, np.ones((5, 5), np.uint8))
        assert core.cut_doodle(inner, h, 0).astype(int).sum() >= inner.astype(int).sum() * 0.99

    # A colored object with no dark ink at all is found by its color
    pic = np.zeros((600, 800, 3), np.uint8)
    pic[:] = (40, 40, 200)
    cv2.circle(pic, (400, 300), 100, (60, 190, 120), -1)
    disc_loop = np.stack([400 + 190 * np.cos(t_), 300 + 190 * np.sin(t_)], 1).astype(np.int32)
    none = np.zeros((600, 800), np.uint8)
    assert core.hug_loop(none, disc_loop) is None
    h = core.hug_loop(none, disc_loop, color=pic)
    r_ = np.hypot(h[:, 0] - 400, h[:, 1] - 300)
    assert 98 < r_.min() and r_.max() < 115, (r_.min(), r_.max())

    # Smoothing: rounder, fewer corners, never loses ink, stays in the loop
    jag = np.zeros((900, 1200), np.uint8)
    for k in range(9):                               # a spiky "sun" of separate strokes
        a = k * 2 * np.pi / 9
        p0 = (int(400 + 60 * np.cos(a)), int(400 + 60 * np.sin(a)))
        p1 = (int(400 + 150 * np.cos(a)), int(400 + 150 * np.sin(a)))
        cv2.line(jag, p0, p1, 255, 6)
    cv2.circle(jag, (400, 400), 40, 255, 6)
    rough, smooth = core.hug_loop(jag, loop, smooth=0), core.hug_loop(jag, loop, smooth=100)
    perim = lambda p: cv2.arcLength(p.reshape(-1, 1, 2), True)
    assert len(smooth) < len(rough), (len(smooth), len(rough))
    assert perim(smooth) / area(smooth) < perim(rough) / area(rough)
    full = core.cut_doodle(jag, loop, 0).astype(int).sum()
    assert abs(core.cut_doodle(jag, smooth, 0).astype(int).sum() - full) <= full * 0.01
    assert all(cv2.pointPolygonTest(lp, (float(x), float(y)), True) >= -2 for x, y in smooth)
    t = time.perf_counter()
    core.hug_loop(np.pad(jag, ((0, 2220), (0, 2960))), loop * 3, smooth=100)  # photo-sized
    ms_smooth = (time.perf_counter() - t) * 1000

    # GUI: on by default, a loose loop hugs; off keeps the loop as drawn
    class Ev:
        def __init__(self, x, y):
            self.x, self.y = int(round(x)), int(round(y))

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "photo.png"
        core.save_png(src, fake_photo()[0])
        root = tk.Tk()
        root.geometry("1300x800")
        app = DoodleApp(root)
        root.update()
        app.open_path(src)
        root.update()
        assert not app.hug.get() and not app.detect_paper.get(), "every switch starts off"
        assert app.white.get() == 255
        app.hug.set(True)              # these tests are about hugging...
        app.detect_paper.set(True)     # ...on a photo with paper detection on
        app.reprepare()
        # White 255 brings back the faint grid; hugging must not treat it as drawing
        app.white.set(160)
        app.refresh()
        tight = core.hug_loop(app._hug_alpha(), loop, smooth=30)
        app.white.set(255)
        app.refresh()
        capped = core.make_alpha(app.flat, app.mask, app.black.get(), 160)
        assert np.array_equal(app._hug_alpha(), capped), "hugging caps White at 160"
        assert np.array_equal(core.hug_loop(app._hug_alpha(), loop, smooth=30), tight)

        def lasso(points):
            pts = [app.to_canvas(x, y) for x, y in points]
            app._lasso_start(Ev(*pts[0]))
            for p in pts[1:]:
                app._lasso_move(Ev(*p))
            app._lasso_end(Ev(*pts[-1]))

        lasso(circle_loop())
        poly = app.doodle_poly(0)
        assert area(poly) < area(loop) * 0.8 and "hugging" in app.status.get()
        # Smoothing re-hugs the selected doodle live
        assert app.doodles[0].hugged and app.doodles[0].smooth == app.smooth.get()
        app.smooth.set(100)
        app.resmooth()
        assert app.doodles[0].smooth == 100
        drawn_loop = core.transform_points(app.R, app.doodles[0].loop).round().astype(np.int32)
        want = core.hug_loop(app.alpha, drawn_loop, smooth=100)
        assert np.abs(app.doodle_poly(0) - want).max() <= 1, "re-hugged from the drawn loop"
        assert "smoothing" in app.status.get()

        app.hug.set(False)
        lasso(circle_loop())
        assert area(app.doodle_poly(1)) > area(loop) * 0.9
        drawn = app.doodle_poly(1).copy()
        app.smooth.set(0)
        app.resmooth()                          # not hugged: left alone
        assert np.array_equal(app.doodle_poly(1), drawn)
        app.select_doodle(0)                     # slider shows the doodle's own value
        assert app.smooth.get() == 100
        root.destroy()
    print(f"hug loop ok ({ms:.0f} ms, smoothing on a photo-sized loop {ms_smooth:.0f} ms)")


def test_pick_object():
    import tkinter as tk
    from doodle_gui import DoodleApp

    area = lambda p: cv2.contourArea(p.reshape(-1, 1, 2).astype(np.float32))
    covers = lambda mask, poly: (core.cut_doodle(mask, poly, 0).astype(int).sum()
                                 >= mask.astype(int).sum() * 0.99)

    # Ink only (pencil-like): clicking the ring picks the ring, not the far line
    ink = np.zeros((900, 1200), np.uint8)
    cv2.circle(ink, (400, 400), 150, 255, 6)
    cv2.line(ink, (800, 200), (1100, 700), 255, 6)
    h = core.pick_object(ink, None, (550, 400), 50)
    ring = np.zeros_like(ink)
    cv2.circle(ring, (400, 400), 150, 255, 6)
    line = ink & ~ring
    assert covers(ring, h) and core.cut_doodle(line, h, 0).max() == 0
    assert core.pick_object(ink, None, (600, 820), 50) is None, "nothing near: no doodle"
    near = core.pick_object(ink, None, (553, 400), 50)            # a few px off the line
    assert near is not None and covers(ring, near)

    # Reach: two strokes 30 px apart
    two = np.zeros((900, 1200), np.uint8)
    cv2.line(two, (300, 300), (300, 600), 255, 6)
    cv2.line(two, (330, 300), (330, 600), 255, 6)
    left = np.zeros_like(two)
    cv2.line(left, (300, 300), (300, 600), 255, 6)
    h0 = core.pick_object(two, None, (300, 450), 0)
    h100 = core.pick_object(two, None, (300, 450), 100)
    assert core.cut_doodle(two & ~left, h0, 0).max() == 0, "reach 0: one stroke"
    assert covers(two, h100), "reach 100: both strokes"

    # A solid W on red: click inside it, on its edge, or in the red next to it
    wpic = np.zeros((600, 800, 3), np.uint8)
    wpic[:] = (40, 40, 200)
    wmask = np.zeros((600, 800), np.uint8)
    cv2.polylines(wmask, [np.array([(200, 150), (290, 450), (400, 220), (510, 450), (600, 150)])],
                  False, 255, 60)
    wpic[wmask > 0] = (30, 30, 30)
    faint = np.zeros((600, 800), np.uint8)       # ink detection sees only the edges
    faint[cv2.morphologyEx(wmask, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0] = 120
    inner = cv2.erode(wmask, np.ones((5, 5), np.uint8))
    for seed in ((245, 300), (215, 300), (206, 300)):   # inside, edge, just outside
        h = core.pick_object(faint, wpic, seed, 50)
        assert h is not None, seed
        poly = h.reshape(-1, 1, 2)
        for valley in ((290, 250), (510, 250), (400, 420)):
            assert cv2.pointPolygonTest(poly, valley, False) < 0, (seed, valley)
        assert covers(inner, h), seed
    assert core.pick_object(faint, wpic, (60, 560), 50) is None, "background: nothing"

    # A hole: clicking inside an O picks the O
    opic = np.zeros((600, 800, 3), np.uint8)
    opic[:] = (40, 40, 200)
    omask = np.zeros((600, 800), np.uint8)
    cv2.circle(omask, (400, 300), 120, 255, 50)
    opic[omask > 0] = (30, 30, 30)
    h = core.pick_object(np.zeros((600, 800), np.uint8), opic, (400, 300), 50)
    assert h is not None and covers(cv2.erode(omask, np.ones((5, 5), np.uint8)), h)

    # GUI: the Auto select doodle button, clicks, drags, Reach
    class Ev:
        def __init__(self, x, y):
            self.x, self.y = int(round(x)), int(round(y))

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "photo.png"
        core.save_png(src, fake_photo()[0])
        root = tk.Tk()
        root.geometry("1300x800")
        app = DoodleApp(root)
        root.update()
        app.open_path(src)
        root.update()
        app.set_select_mode(True)
        app.pick_button.invoke()
        assert app.pick_mode.get() and not app.select_mode.get()
        assert "pressed" in app.pick_button.state() and str(app.canvas.cget("cursor")) == "tcross"
        app.set_select_mode(True)                  # the modes switch each other off
        assert not app.pick_mode.get()
        app.set_pick_mode(True)

        def click(x, y, dx=0, dy=0):
            a = app.to_canvas(x, y)
            app._left_press(Ev(*a))
            app._left_move(Ev(a[0] + dx, a[1] + dy))
            app._left_release(Ev(a[0] + dx, a[1] + dy))

        click(550, 400)                            # on the ring
        assert len(app.doodles) == 1 and app.doodles[0].seed is not None
        assert "picked" in app.status.get()
        ring_poly = app.doodle_poly(0)
        assert cv2.pointPolygonTest(ring_poly.reshape(-1, 1, 2), (400.0, 400.0), False) > 0
        assert cv2.pointPolygonTest(ring_poly.reshape(-1, 1, 2), (800.0, 425.0), False) < 0
        click(300, 400, 80, 0)                     # a drag pans, no doodle
        assert len(app.doodles) == 1
        click(100, 870)                            # off the page: nothing to pick
        assert len(app.doodles) == 1 and "Nothing to pick" in app.status.get()

        # Reach re-picks the selected doodle; selecting restores its values
        app.reach.set(100)
        app.repick()
        assert app.doodles[0].reach == 100
        app.smooth.set(80)
        app.resmooth()
        assert app.doodles[0].smooth == 80
        app.reach.set(10)
        app.smooth.set(10)
        app.select_doodle(0)
        assert app.reach.get() == 100 and app.smooth.get() == 80

        app._on_escape()
        assert not app.pick_mode.get() and "pressed" not in app.pick_button.state()
        root.destroy()
    print("pick object ok")


def test_combine_shapes():
    size = (400, 600)
    sq = lambda x0, y0, x1, y1: np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], np.int32)
    ink = np.full(size, 255, np.uint8)

    both = core.combine_shapes(sq(50, 50, 150, 150), sq(300, 50, 400, 150), "add", size)
    assert len(both) == 2
    cut = core.cut_doodle(ink, both, 0)
    assert cut.shape[1] >= 350 and cut.astype(int).sum() >= 2 * 100 * 100 * 255 * 0.97
    assert core.shape_contains(both, (100, 100)) and core.shape_contains(both, (350, 100))
    assert not core.shape_contains(both, (220, 100))

    holed = core.combine_shapes(sq(50, 50, 250, 250), sq(100, 100, 200, 200), "cut", size)
    assert len(holed) == 2, "outline plus hole"
    assert not core.shape_contains(holed, (150, 150)) and core.shape_contains(holed, (70, 70))
    out = core.cut_doodle(ink, holed, 0)        # cropped at (50, 50)
    assert out[100, 100] == 0 and out[20, 20] == 255, "hole dropped, ring kept"
    src, _ = core.doodle_source(255 - ink, None, holed, margin=0)
    assert src[100, 100] == 255 and src[20, 20] == 0, "doodle editor sees the hole as paper"

    assert core.combine_shapes(sq(100, 100, 200, 200), sq(50, 50, 250, 250), "cut", size) is None
    print("combine shapes ok")


def test_doodle_add_cut_gui():
    import tkinter as tk
    from tkinter import ttk
    from doodle_gui import DoodleApp

    class Ev:
        def __init__(self, x, y):
            self.x, self.y = int(round(x)), int(round(y))

    bgr = fake_photo()[0]
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "photo.png"
        core.save_png(src, bgr)
        root = tk.Tk()
        root.geometry("1300x800")
        app = DoodleApp(root)
        root.update()
        app.open_path(src)
        app.white.set(160)
        app.set_extract(True)
        app.set_extract_open(True)    # the slider checks below need its settings visible
        app.hug.set(False)            # loops kept exactly as drawn: predictable shapes
        root.update()

        def left_drag(points):
            pts = [app.to_canvas(x, y) for x, y in points]
            app._left_press(Ev(*pts[0]))
            for p in pts[1:]:
                app._left_move(Ev(*p))
            app._left_release(Ev(*pts[-1]))

        app.set_select_mode(True)
        left_drag(circle_loop())                                   # doodle 1: the circle
        left_drag([(620, 220), (980, 220), (980, 630), (620, 630)])  # doodle 2: the line
        app.set_select_mode(False)
        inside = lambda i, pt: core.shape_contains(app.doodle_poly(i), pt)

        # Rows: small + / − under the thumbnail; the long buttons are gone
        texts = lambda w: [c.cget("text") for c in w.winfo_children()
                           if isinstance(c, (tk.Button, ttk.Button))]
        assert texts(app._rows[0][6]) == ["+", "−"]
        assert texts(app._rows[0][4]) == ["Open…", "Save…", "✕ Remove"]

        # Thumbnails match the background: a checkerboard where it's transparent...
        app.bg_mode.set("Transparent")
        rgba = app._doodle_rgba(0)
        thumb = app._preview_rgb(rgba)
        clear = rgba[:, :, 3] == 0
        assert set(np.unique(thumb[clear])) <= {150, 210} and len(np.unique(thumb[clear])) == 2
        # ...and the background and line color are saved into the image
        app.bg_mode.set("Black")
        app.tint = (255, 0, 0)
        rgba = app._doodle_rgba(0)
        assert (rgba[:, :, 3] == 255).all(), "a solid background makes it opaque"
        assert (rgba[clear][:, :3] == 0).all(), "black where there's no ink"
        solid = cv2.erode((~clear).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ink_px = rgba[solid & (app._doodle_rgba(0)[:, :, 2] > 200)]
        assert len(ink_px) and (ink_px[:, 0] < 60).all(), "red lines (saved as BGR)"
        thumb = app._preview_rgb(rgba)
        assert (thumb[clear] == 0).all(), "the thumbnail shows the black background"
        app.bg_mode.set("White")                        # the photo gets it too
        app.set_extract(False)
        photo = app._doodle_rgba(0)
        assert (photo[:, :, 3] == 255).all() and (photo[0, 0, :3] == 255).all()
        app.set_extract(True)
        # Custom…: OK picks the color, Cancel goes back to the previous background
        import doodle_gui
        answers = iter([((10, 200, 30), "#0ac81e"), (None, None)])
        real = doodle_gui.colorchooser.askcolor
        doodle_gui.colorchooser.askcolor = lambda **kw: next(answers)
        try:
            app.bg_mode.set("Custom…")
            app.on_background()
            assert app.bg_custom == (10, 200, 30) and app._bg_rgb() == (10, 200, 30)
            assert app.bg_swatch.winfo_manager()
            app.bg_mode.set("Black")
            app.on_background()
            app.bg_mode.set("Custom…")
            app.on_background()                         # cancelled
            assert app.bg_mode.get() == "Black" and not app.bg_swatch.winfo_manager()
        finally:
            doodle_gui.colorchooser.askcolor = real
        editor = app.edit_doodle(0)                      # editors save the same way
        assert editor.tint == (255, 0, 0) and editor.bg_mode.get() == "Black"
        editor.root.destroy()
        app.bg_mode.set("Transparent")
        app.on_background()
        app.tint = (255, 255, 255)

        # Banner over the preview names the mode
        banner = lambda: app.canvas.find_withtag("banner")
        assert app._mode_banner_text() is None and not banner()
        app.set_pick_mode(True)
        assert app._mode_banner_text() == "Auto selecting doodles (Esc to stop)" and banner()
        app.set_select_mode(True)
        assert app._mode_banner_text() == "Lassoing doodles (Esc to stop)" and banner()
        app.set_select_mode(False)
        assert not banner()
        app.start_crop()
        assert app._mode_banner_text().startswith("Cropping") and banner()
        app.save_crop()
        assert not banner()

        # Hovering a button shows its hint in the status bar, and leaving puts the
        # old message back -- unless clicking the button put a new one there
        app.status.set("before")
        app.select_button.event_generate("<Enter>")
        assert app.status.get() == "Drag a loop around a drawing to make it a doodle"
        app.select_button.event_generate("<Leave>")
        assert app.status.get() == "before"
        app.select_button.event_generate("<Enter>")
        app.select_button.invoke()
        app.select_button.event_generate("<Leave>")
        assert app.select_mode.get() and app.status.get().startswith("Lasso doodle:")
        app.set_select_mode(False)

        # Slider − / + buttons step one notch and act like dragging; sliders have hints
        b0 = app.black.get()
        assert app.black_scale.step(1) and app.black.get() == b0 + 1
        assert app._refresh_job is not None, "stepping Black recomputes the result"
        assert app.black_scale.step(-1) and app.black.get() == b0
        app.white.set(255)
        assert not app.white_scale.step(1) and app.white.get() == 255, "stops at the end"
        app.white.set(160)
        app.on_white()
        # Extract ink / Crop / Rotate share one row; Rotate opens its settings
        row = (app.extract_button, app.crop_button, app.rotate_button)
        assert len({b.winfo_parent() for b in row}) == 1
        assert len({b.winfo_y() for b in row}) == 1
        assert not app.rotate_section.winfo_manager()
        assert "pressed" not in app.rotate_button.state()
        app.rotate_button.invoke()
        assert app.rotate_section.winfo_manager() and "pressed" in app.rotate_button.state()
        turn_right = [w for w in app.rotate_section.winfo_children()[0].winfo_children()
                      if w.cget("text") == "⟳ 90°"][0]
        turn_right.invoke()
        assert app.quarter == 1
        app.rotate_button.invoke()
        assert not app.rotate_section.winfo_manager() and app.quarter == 1, "rotation stays"

        # Only one of Extract ink / Crop / Rotate is open at a time
        opened = lambda: [n for n, on in (("extract", app.extract_section.winfo_manager()),
                                          ("crop", app.cropping),
                                          ("rotate", app.rotate_section.winfo_manager())) if on]
        pressed = lambda: [b.cget("text") for b in row if "pressed" in b.state()]
        app.extract_button.invoke()
        assert opened() == ["extract"] and len(pressed()) == 1
        app.crop_button.invoke()
        assert opened() == ["crop"] and pressed() == ["Crop"]
        app.rotate_button.invoke()
        assert opened() == ["rotate"] and pressed() == ["Rotate"]
        assert app.crop_corners is None, "closing Crop this way keeps (here: no) crop"
        app.extract_button.invoke()
        assert opened() == ["extract"]
        app.extract_button.invoke()
        assert opened() == [] and pressed() == []
        app.rotate_photo(-1)
        root.update()
        assert app.straighten_scale.step(1) and app.angle.get() == 0.5
        assert app._rotate_job is not None
        app.straighten_scale.step(-1)
        assert app.angle.get() == 0.0
        app.refresh()
        app.status.set("before")
        app.white_scale.event_generate("<Enter>")
        assert app.status.get().startswith("Anything this light")
        app.white_scale.event_generate("<Leave>")
        assert app.status.get() == "before"
        minus, plus = app.white_scale.step_buttons
        assert minus.cget("text") == "−" and plus.cget("text") == "+"
        root.update()
        sc = app.white_scale
        track = int(sc.cget("width")) + 2 * int(sc.cget("borderwidth"))
        track_bottom = sc.winfo_rooty() + sc.winfo_height() - int(sc.cget("highlightthickness"))
        for b in (minus, plus):
            assert abs(b.winfo_height() - track) <= 1, (b.winfo_height(), track)
            assert abs(b.winfo_rooty() + b.winfo_height() - track_bottom) <= 2

        # Doodle settings slot above Auto select doodle: fixed size, only what matters
        root.update()
        shown = lambda: [k for k, f in app._opt_frames.items() if f.winfo_ismapped()]
        slot_h, list_h = app.opts_frame.winfo_height(), app.list_canvas.winfo_height()
        assert shown() == [] and slot_h > 50
        for mode, want in (("select", ["hug", "smooth"]), ("pick", ["reach", "smooth"]),
                           ("cut", []), ("add", ["hug", "smooth"]), (None, [])):
            app.set_select_mode(mode == "select")
            app.set_pick_mode(mode == "pick")
            app.set_edit_mode((0, mode) if mode in ("cut", "add") else None)
            root.update()
            assert shown() == want, (mode, shown())
            assert str(app.opts_frame.cget("style")) == ("Opts.TFrame" if want else "TFrame")
            assert app.opts_frame.winfo_height() == slot_h, "the slot keeps its size"
            assert app.list_canvas.winfo_height() == list_h, "the list doesn't jump"
            for k in want:                               # everything shown fits
                f = app._opt_frames[k]
                assert f.winfo_y() + f.winfo_height() <= slot_h, (mode, k)
        app.set_select_mode(True)
        root.update()
        s0 = app.smooth.get()
        assert app.smooth_scale.step(1) and app.smooth.get() == s0 + 1
        assert app._smooth_job is not None
        app.smooth_scale.step(-1)
        app.status.set("before")
        app.smooth_scale.event_generate("<Enter>")
        assert app.status.get().startswith("How much hugged outlines")
        app.smooth_scale.event_generate("<Leave>")
        assert app.status.get() == "before"
        app.set_select_mode(False)

        # + Add area: a loop elsewhere joins the doodle as a second outline
        app._rows[0][6].winfo_children()[0].invoke()
        assert app.edit_mode == (0, "add") and app.selected == 0
        assert "pressed" in app._rows[0][6].winfo_children()[0].state()
        assert app._mode_banner_text().startswith("Adding to doodle 1") and banner()
        assert str(app.canvas.cget("cursor")) == "crosshair" and not app.select_mode.get()
        left_drag([(250, 650), (350, 650), (350, 750), (250, 750)])
        assert len(app.doodles) == 2 and app.doodles[0].edited
        assert inside(0, (300, 700)) and inside(0, (400, 260))
        assert app.edit_mode == (0, "add"), "mode stays on"

        # Click inside doodle 2: its whole area is added, and doodle 2 stays
        left_drag([(800, 425)])
        assert inside(0, (800, 425)) and inside(0, (630, 230)) and len(app.doodles) == 2
        rgba = app._doodle_rgba(0)
        assert rgba.shape[1] > 600, "the saved doodle spans both drawings"
        # A click on nothing changes nothing
        before = [c.copy() for c in app.doodles[0].poly]
        left_drag([(150, 150)])
        assert all(np.array_equal(a, b) for a, b in zip(before, app.doodles[0].poly))

        # − Cut area: a hole in the circle's middle
        app.toggle_edit_mode(0, "cut")
        assert app.edit_mode == (0, "cut")
        left_drag([(330, 330), (470, 330), (470, 470), (330, 470)])
        assert not inside(0, (400, 400)) and inside(0, (400, 260))
        # Cutting everything is refused
        before = [c.copy() for c in app.doodles[0].poly]
        left_drag([(20, 20), (1180, 20), (1180, 880), (20, 880)])
        assert len(app.doodles[0].poly) == len(before)

        # Smoothing leaves an edited doodle alone
        app.smooth.set(90)
        app.resmooth()
        assert all(np.array_equal(a, b) for a, b in zip(before, app.doodles[0].poly))

        # Rotation keeps the edited shape attached; the editor window opens
        app.rotate_photo(1)
        root.update()
        editor = app.edit_doodle(0)
        assert editor is not None
        editor.root.destroy()
        app.rotate_photo(-1)
        root.update()
        assert not inside(0, (400, 400)) and inside(0, (400, 260)) and inside(0, (300, 700))

        # Esc stops; clicking the pressed button again also stops
        app._on_escape()
        assert app.edit_mode is None and str(app.canvas.cget("cursor")) == "fleur"
        assert not banner()
        app.toggle_edit_mode(1, "add")
        app.toggle_edit_mode(1, "add")
        assert app.edit_mode is None

        # Deleting doodles keeps the target index right
        app.toggle_edit_mode(1, "cut")
        app.delete_doodle(0)
        assert app.edit_mode == (0, "cut")
        app.delete_doodle(0)
        assert app.edit_mode is None

        # The doodles list doesn't scroll when every card fits
        lc = app.list_canvas
        app.add_doodle(circle_loop())
        root.update()
        y0 = app._rows[0][0].winfo_rooty()
        for n in (-5, 5, -5):
            lc.yview_scroll(n, "units")
        root.update()
        assert lc.yview() == (0.0, 1.0) and app._rows[0][0].winfo_rooty() == y0
        for _ in range(8):                      # too many to fit: now it scrolls
            app.add_doodle(circle_loop())
        root.update()
        lc.yview_scroll(5, "units")
        assert lc.yview()[0] > 0
        app.clear_doodles()
        root.update()
        assert lc.yview()[0] == 0.0, "back at the top once the list fits again"
        root.destroy()
    print("doodle add/cut ok")


if __name__ == "__main__":
    test_pipeline()
    test_doodles()
    test_unicode_path()
    test_gui()
    test_rotation_core()
    test_doodle_source()
    test_rotation_gui()
    test_edit_doodle_window()
    test_crop_tool()
    test_crop_to_paper()
    test_mouse_modes()
    test_extract_toggle()
    test_theme()
    test_hug_loop()
    test_pick_object()
    test_combine_shapes()
    test_doodle_add_cut_gui()
    test_preview_keeps_thin_lines()
    print("all tests passed")
