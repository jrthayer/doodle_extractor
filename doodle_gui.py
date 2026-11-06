"""Doodle Converter: windowed app.

Open a photo of a drawing, tune the sliders while watching the preview,
then save a Godot-ready PNG (white lines, transparent everywhere else).
Drag around parts of the photo to cut them out as separate pieces.
Scroll to zoom, right-drag to pan.
"""

import math
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import doodle_core as core

IMAGE_TYPES = [("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
               ("All files", "*.*")]
BACKGROUNDS = ("Light", "Dark", "Checker")
THUMB = 96                    # piece thumbnail size (px)
MAX_ZOOM = 8.0                # 800%: screen px per photo px
ZOOM_STEP = 1.25
PIECE_COLOR = "#1e90ff"
SELECTED_COLOR = "#ff8c00"
CROP_COLOR = "#ff3b6b"
ROW_BG, ROW_SELECTED_BG = "#f4f4f4", "#ffe2bf"


@dataclass
class Piece:
    poly: np.ndarray      # loop in ORIGINAL photo coords (float), so it follows photo rotation


_APPS = []                # every open converter window (main + piece editors)


def _dispatch_wheel(e):
    """The mouse wheel is bound once for the whole program; route it to
    whichever window's preview or list is under the pointer."""
    for app in list(_APPS):
        if app._wheel(e):
            return


class DoodleApp:
    def __init__(self, root):
        self.root = root      # a tk.Tk (main window) or tk.Toplevel (piece editor)
        root.title("Doodle Converter")
        root.geometry("1350x800")
        root.minsize(900, 500)

        self.path = None          # current photo
        self.base = None          # file-name stem for saving
        self.is_piece = False     # True in a piece-editor window
        self.bgr = None           # full-res photo
        self.flat0 = None         # full-res prepared image, unrotated
        self.mask0 = None
        self.rflat = None         # ... after photo rotation (full size)
        self.rmask = None
        self.alpha_full = None    # result for the current settings, full rotated size
        self.quarter = 0          # photo: 90-degree clockwise turns
        self.Rrot = np.eye(3)     # original photo coords -> rotated coords
        self._rotate_job = None

        # The "working" image is the rotated photo cut to the saved crop (or the
        # whole rotated photo while cropping). Previews, pieces and saving use it.
        self.flat = self.mask = self.alpha = None
        self.work_rect = None     # (x, y, w, h) of the working image in rotated coords
        self.R = np.eye(3)        # original photo coords -> working coords
        self.Rinv = np.eye(3)

        # Crop: saved as corners in ORIGINAL photo coords so it follows rotation
        self.crop_corners = None  # None = no crop
        self.crop_fit_saved = False   # saved crop tracks the drawing's ink box
        self.cropping = False     # crop mode active
        self.edit_corners = None  # crop box being edited (original photo coords)
        self._fit_backup = None   # box to restore when "Crop to the drawing" is unticked
        self._crop_drag = None

        # View: `scale` screen px per photo px; (vx, vy) = photo coords at the
        # canvas's top-left corner. `fit` = follow the window size.
        self.scale = 1.0
        self.vx = self.vy = 0.0
        self.fit = True
        self._pan = None

        self.tint = (0, 0, 0)     # preview line color (RGB)
        self._photo = None        # keep Tk images alive
        self._refresh_job = None
        self._resize_job = None
        self._bg_cache = {}

        self.pieces = []          # list of Piece
        self.selected = None      # index into self.pieces
        self._lasso = None        # photo coords of the loop being drawn
        self._lasso_last = None   # last lasso point in canvas coords
        self._rows = []           # (frame, thumb label, text label) per piece
        self._thumbs = []

        self.black = tk.IntVar(value=80)
        self.white = tk.IntVar(value=160)
        self.min_speck = tk.IntVar(value=0)
        self.pad = tk.IntVar(value=0)    # margin kept around drawings (no longer a slider)
        self.crop_fit = tk.BooleanVar(value=False)   # "Crop to the drawing" in crop mode
        self.crop_vars = {k: tk.StringVar(value="0") for k in ("x", "y", "w", "h")}
        self.detect_paper = tk.BooleanVar(value=True)
        self.bg_mode = tk.StringVar(value="Light")
        self.zoom_text = tk.StringVar(value="")
        self.angle = tk.DoubleVar(value=0.0)        # photo straighten, degrees clockwise

        self._build_ui()
        root.bind("<Control-o>", lambda e: self.open_dialog())
        root.bind("<Control-s>", lambda e: self.save_dialog())
        root.bind("<Control-0>", lambda e: self.zoom_fit())
        root.bind("<Control-equal>", lambda e: self.zoom_step(ZOOM_STEP))
        root.bind("<Control-plus>", lambda e: self.zoom_step(ZOOM_STEP))
        root.bind("<Control-minus>", lambda e: self.zoom_step(1 / ZOOM_STEP))
        root.bind("<Escape>", self._on_escape)
        root.bind("<Return>", self._on_return)
        root.bind("<Delete>", self._on_delete)
        _APPS.append(self)
        root.bind_all("<MouseWheel>", _dispatch_wheel)
        root.bind("<Destroy>", self._on_destroy)

    def _on_delete(self, _e=None):
        """Delete key removes the selected piece, unless you're typing in a box."""
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry)):
            return
        if self.selected is not None:
            self.delete_piece(self.selected)

    def _on_destroy(self, e):
        if e.widget is self.root and self in _APPS:
            _APPS.remove(self)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="Open…", command=self.open_dialog).pack(side="left")
        ttk.Button(bar, text="Save PNG…", command=self.save_dialog).pack(side="left", padx=(6, 18))

        ttk.Label(bar, text="Preview background:").pack(side="left")
        bg = ttk.Combobox(bar, textvariable=self.bg_mode, values=BACKGROUNDS,
                          state="readonly", width=8)
        bg.pack(side="left", padx=(4, 12))
        bg.bind("<<ComboboxSelected>>", lambda e: self.schedule_refresh())

        ttk.Label(bar, text="Preview line color:").pack(side="left")
        self.tint_swatch = tk.Button(bar, width=3, relief="solid", bd=1,
                                     bg=self._hex(self.tint), command=self.pick_tint)
        self.tint_swatch.pack(side="left", padx=(4, 18))

        ttk.Label(bar, text="Zoom:").pack(side="left")
        ttk.Button(bar, text="−", width=3,
                   command=lambda: self.zoom_step(1 / ZOOM_STEP)).pack(side="left", padx=(4, 0))
        ttk.Label(bar, textvariable=self.zoom_text, width=6, anchor="center").pack(side="left")
        ttk.Button(bar, text="+", width=3,
                   command=lambda: self.zoom_step(ZOOM_STEP)).pack(side="left")
        ttk.Button(bar, text="Fit", width=4, command=self.zoom_fit).pack(side="left", padx=4)
        ttk.Button(bar, text="100%", width=5,
                   command=lambda: self.zoom_to(1.0)).pack(side="left")

        self.status = tk.StringVar(value="Open a photo to start (Ctrl+O).")
        ttk.Label(self.root, textvariable=self.status, anchor="w",
                  padding=(8, 3), relief="sunken").pack(side="bottom", fill="x")

        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True)

        self._build_settings(body)
        self._build_pieces_panel(body)

        self.canvas = tk.Canvas(body, bg="#808080", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_resize)
        # Left button: crop box editing in crop mode, otherwise the lasso
        self.canvas.bind("<ButtonPress-1>", lambda e: (self._crop_press if self.cropping
                                                       else self._lasso_start)(e))
        self.canvas.bind("<B1-Motion>", lambda e: (self._crop_move if self.cropping
                                                   else self._lasso_move)(e))
        self.canvas.bind("<ButtonRelease-1>", lambda e: (self._crop_release if self.cropping
                                                         else self._lasso_end)(e))
        self.canvas.bind("<Motion>", self._crop_hover)
        for b in (2, 3):  # middle or right drag pans
            self.canvas.bind(f"<ButtonPress-{b}>", self._pan_start)
            self.canvas.bind(f"<B{b}-Motion>", self._pan_move)
            self.canvas.bind(f"<ButtonRelease-{b}>", self._pan_end)
        self._show_hint()

    def _build_settings(self, parent):
        side = ttk.Frame(parent, padding=(10, 6))
        side.pack(side="left", fill="y")
        self._slider(side, "Black (solid ink at or below)", self.black, 0, 255, self.on_black)
        self._slider(side, "White (transparent at or above)", self.white, 0, 255, self.on_white)
        self._slider(side, "Remove specks smaller than (px)", self.min_speck, 0, 2000,
                     lambda v: self.schedule_refresh())

        ttk.Separator(side).pack(fill="x", pady=10)
        self.paper_check = ttk.Checkbutton(side, text="Detect paper (ignore the desk)",
                                           variable=self.detect_paper, command=self.reprepare)
        self.paper_check.pack(anchor="w")

        ttk.Separator(side).pack(fill="x", pady=10)
        self.crop_button = ttk.Button(side, text="Crop…", command=self.start_crop)
        self.crop_button.pack(anchor="w")
        self.crop_holder = ttk.Frame(side)   # keeps the crop section's place in the column
        self.crop_holder.pack(fill="x")
        self._build_crop_section(self.crop_holder)

        ttk.Separator(side).pack(fill="x", pady=10)
        ttk.Label(side, text="Rotate photo").pack(anchor="w")
        row = ttk.Frame(side)
        row.pack(anchor="w", pady=(4, 0))
        ttk.Button(row, text="⟲ 90°", command=lambda: self.rotate_photo(-1)).pack(side="left")
        ttk.Button(row, text="⟳ 90°", command=lambda: self.rotate_photo(1)).pack(side="left", padx=4)
        ttk.Button(row, text="Reset", command=self.reset_rotation).pack(side="left")
        self._slider(side, "Straighten (degrees clockwise)", self.angle, -45, 45,
                     lambda v: self.schedule_rotate(), resolution=0.5)

        ttk.Separator(side).pack(fill="x", pady=10)
        ttk.Button(side, text="Reset settings", command=self.reset).pack(anchor="w")
        tip = ("• Grid showing? Lower White.\n"
               "• Lines broken? Raise White.\n"
               "• Line middles gray? Raise Black.\n"
               "• Scroll to zoom, right-drag to pan.\n"
               "• Color it in Godot: Self Modulate.")
        self.tips = ttk.Label(side, text=tip, foreground="#555", justify="left")
        self.tips.pack(anchor="w", pady=(10, 0))

    def _build_crop_section(self, parent):
        """The crop settings, shown only while cropping."""
        sec = self.crop_section = ttk.LabelFrame(parent, text="Crop", padding=8)
        ttk.Checkbutton(sec, text="Crop to the drawing", variable=self.crop_fit,
                        command=self.on_crop_fit).grid(row=0, column=0, columnspan=4,
                                                       sticky="w", pady=(0, 6))
        self.crop_spins = {}
        for i, (key, label) in enumerate((("x", "X"), ("y", "Y"), ("w", "Width"), ("h", "Height"))):
            r, c = divmod(i, 2)
            ttk.Label(sec, text=label).grid(row=1 + r, column=2 * c, sticky="w",
                                            padx=(0 if c == 0 else 10, 4), pady=2)
            sp = ttk.Spinbox(sec, from_=0, to=99999, increment=1, width=7,
                             textvariable=self.crop_vars[key], command=self._crop_from_spins)
            sp.grid(row=1 + r, column=2 * c + 1, sticky="w", pady=2)
            sp.bind("<Return>", lambda e: self._crop_from_spins() or "break")
            sp.bind("<FocusOut>", lambda e: self._crop_from_spins())
            self.crop_spins[key] = sp
        btns = ttk.Frame(sec)
        btns.grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Button(btns, text="Save crop", command=self.save_crop).pack(side="left")
        ttk.Button(btns, text="Cancel", command=self.cancel_crop).pack(side="left", padx=6)

    def _build_pieces_panel(self, parent):
        panel = ttk.Frame(parent, padding=(6, 6))
        panel.pack(side="right", fill="y")
        ttk.Label(panel, text="Pieces", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(panel, text="Drag around a drawing on the\npreview to cut it out.\n"
                              "Esc cancels, Delete removes.",
                  foreground="#555", justify="left").pack(anchor="w")

        btns = ttk.Frame(panel)
        btns.pack(side="bottom", fill="x", pady=(6, 0))
        ttk.Button(btns, text="Save all…", command=self.save_all_dialog).pack(side="left")
        ttk.Button(btns, text="Clear", command=self.clear_dialog).pack(side="left", padx=6)

        wrap = ttk.Frame(panel)
        wrap.pack(fill="both", expand=True, pady=(6, 0))
        self.list_canvas = tk.Canvas(wrap, width=250, highlightthickness=0)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.list_canvas.yview)
        self.list_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.list_canvas.pack(side="left", fill="both", expand=True)
        self.list_inner = tk.Frame(self.list_canvas)
        self.list_canvas.create_window((0, 0), window=self.list_inner, anchor="nw")
        self.list_inner.bind("<Configure>", lambda e: self.list_canvas.configure(
            scrollregion=self.list_canvas.bbox("all")))

    def _wheel(self, e):
        """Mouse wheel: zoom over the preview, scroll over the pieces list.
        Returns True if the pointer was over this window's widgets."""
        try:
            w = self.root.winfo_containing(e.x_root, e.y_root)
        except (tk.TclError, KeyError):
            return False
        if w is self.canvas:
            factor = ZOOM_STEP if e.delta > 0 else 1 / ZOOM_STEP
            self.zoom_at(e.x_root - self.canvas.winfo_rootx(),
                         e.y_root - self.canvas.winfo_rooty(), factor)
            return True
        while w is not None:
            if w is self.list_canvas:
                self.list_canvas.yview_scroll(int(-e.delta / 120), "units")
                return True
            w = w.master
        return False

    def _slider(self, parent, label, var, lo, hi, command, resolution=1):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(8, 0))
        tk.Scale(parent, from_=lo, to=hi, orient="horizontal", variable=var,
                 length=240, command=command, resolution=resolution).pack(anchor="w")

    def _show_hint(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2 or 400, self.canvas.winfo_height() // 2 or 300,
            text="Open a photo of a drawing (Ctrl+O)", fill="white", font=("Segoe UI", 14))

    @staticmethod
    def _hex(rgb):
        return "#%02x%02x%02x" % rgb

    # ------------------------------------------------------------ settings

    def settings(self):
        return core.Settings(black=self.black.get(), white=self.white.get(),
                             min_speck=self.min_speck.get(), pad=self.pad.get())

    def on_black(self, _=None):
        if self.black.get() >= self.white.get():
            self.white.set(min(self.black.get() + 1, 255))
            self.black.set(self.white.get() - 1)
        self.schedule_refresh()

    def on_white(self, _=None):
        if self.white.get() <= self.black.get():
            self.black.set(max(self.white.get() - 1, 0))
            self.white.set(self.black.get() + 1)
        self.schedule_refresh()

    def reset(self):
        d = core.Settings()
        self.black.set(d.black)
        self.white.set(d.white)
        self.min_speck.set(d.min_speck)
        if not self.detect_paper.get() and self.bgr is not None:
            self.detect_paper.set(True)
            self.reprepare()
        self.schedule_refresh()

    def pick_tint(self):
        rgb, _ = colorchooser.askcolor(color=self._hex(self.tint), title="Preview line color",
                                       parent=self.root)
        if rgb:
            self.tint = tuple(int(c) for c in rgb)
            self.tint_swatch.configure(bg=self._hex(self.tint))
            self.schedule_refresh()

    # ---------------------------------------------------------- open / save

    def open_dialog(self):
        if self.pieces and not messagebox.askyesno(
                "Open another photo?", "This clears the current pieces. Continue?",
                parent=self.root):
            return
        initial = str(self.path.parent) if self.path else None
        p = filedialog.askopenfilename(title="Open drawing", filetypes=IMAGE_TYPES,
                                       initialdir=initial, parent=self.root)
        if p:
            self.open_path(p)

    def open_path(self, p):
        p = Path(p)
        try:
            self._busy(f"Loading {p.name}…")
            self.bgr = core.load_image(p)
        except Exception as e:
            self._idle()
            messagebox.showerror("Couldn't open image", f"{p}\n\n{e}", parent=self.root)
            return False
        self.path, self.base, self.is_piece = p, p.stem, False
        self.root.title(f"Doodle Converter: {p.name}")
        self.paper_check.state(["!disabled"])
        self._reset_for_new_image(fit_crop=False)
        self.reprepare()
        return True

    def _reset_for_new_image(self, fit_crop):
        self.clear_pieces()
        self._leave_crop_mode()
        self.crop_corners, self.crop_fit_saved = None, fit_crop
        self.work_rect = None
        self.quarter = 0
        self.angle.set(0.0)
        self.fit = True

    def load_prepared(self, flat, mask, path, base, settings, label):
        """Load an already-prepared image (a piece from another window)."""
        self.bgr = None                     # nothing to re-prepare from
        self.path, self.base, self.is_piece = Path(path), base, True
        self.root.title(f"Doodle Converter: {self.path.name}, {label}")
        self.paper_check.state(["disabled"])
        self._reset_for_new_image(fit_crop=True)   # start cropped to the drawing
        self.black.set(settings.black)
        self.white.set(settings.white)
        self.min_speck.set(settings.min_speck)
        self.pad.set(settings.pad)
        self.flat0, self.mask0 = flat, mask
        h, w = flat.shape
        self.status.set(f"Editing {label} of {self.path.name} ({w}×{h}). "
                        "Save PNG saves this piece.")
        self.apply_rotation()

    def reprepare(self):
        """Run the slow part of the pipeline for the current photo."""
        if self.bgr is None:
            return
        self._busy("Processing…")
        try:
            self.flat0, self.mask0 = core.prepare(self.bgr, self.detect_paper.get())
        finally:
            self._idle()
        h, w = self.flat0.shape
        paper = "page detected" if self.mask0 is not None else "no separate page found"
        self.status.set(f"{self.path.name}: {w}×{h}, {paper}. "
                        "Scroll to zoom, right-drag to pan.")
        self.apply_rotation()

    # ------------------------------------------------------- photo rotation

    def apply_rotation(self, refit=False):
        """Rotate the prepared photo by the current rotation and redraw."""
        self._rotate_job = None
        if self.flat0 is None:
            return
        angle = float(self.angle.get())
        self.rflat, self.rmask = core.rotate_prepared(self.flat0, self.mask0, self.quarter, angle)
        self.Rrot, _ = core.rotation_matrix(self.flat0.shape, self.quarter, angle)
        self.work_rect = None           # size may have changed: re-place the view
        if refit:
            self.fit = True
        if self.cropping:
            self._update_spin_limits()
        self.refresh()

    def rotate_photo(self, turns):
        if self.flat0 is None:
            return
        self.quarter = (self.quarter + turns) % 4
        self.apply_rotation(refit=True)

    def reset_rotation(self):
        if self.flat0 is None:
            return
        self.quarter = 0
        self.angle.set(0.0)
        self.apply_rotation(refit=True)

    def schedule_rotate(self, *_):
        if self._rotate_job:
            self.root.after_cancel(self._rotate_job)
        self._rotate_job = self.root.after(60, self.apply_rotation)

    def _ready(self):
        """Saving and editing need a photo and no crop in progress."""
        if self.flat is None:
            messagebox.showinfo("Nothing to save", "Open a photo first.", parent=self.root)
            return False
        if self.cropping:
            messagebox.showinfo("Finish cropping", "Click Save crop or Cancel first.",
                                parent=self.root)
            return False
        self._flush()
        return True

    def _flush(self):
        """Apply any pending slider change right away."""
        if self._rotate_job:
            self.root.after_cancel(self._rotate_job)
            self.apply_rotation()
        elif self._refresh_job:
            self.root.after_cancel(self._refresh_job)
            self.refresh()

    def save_dialog(self):
        if not self._ready():
            return
        name = f"{self.base}.png" if self.is_piece else f"{self.base}_ink.png"
        p = filedialog.asksaveasfilename(
            title="Save ink PNG", defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialdir=str(self.path.parent), initialfile=name, parent=self.root)
        if p:
            self.save_path(p)

    def save_path(self, p):
        """Save the working image (the cropped area) as a PNG."""
        if not self._ready():
            return False
        return self._write(p, lambda: core.to_rgba(self.alpha))

    def _write(self, p, make_rgba):
        p = Path(p)
        try:
            self._busy("Saving…")
            rgba = make_rgba()
            core.save_png(p, rgba)
        except Exception as e:
            self._idle()
            messagebox.showerror("Couldn't save", f"{p}\n\n{e}", parent=self.root)
            return False
        self._idle()
        self.status.set(f"Saved {p} ({rgba.shape[1]}×{rgba.shape[0]})")
        return True

    def _busy(self, msg):
        self.status.set(msg)
        self.root.config(cursor="watch")
        self.root.update_idletasks()

    def _idle(self):
        self.root.config(cursor="")

    # --------------------------------------------------------------- pieces

    def _piece_name(self, i):
        return f"{self.base}_piece{i + 1}.png"

    def piece_poly(self, i):
        """Piece i's loop in current (rotated) photo coords."""
        return core.transform_points(self.R, self.pieces[i].poly).round().astype(np.int32)

    def add_piece(self, polygon):
        """Add a loop given in current (rotated) photo coords."""
        orig = core.transform_points(self.Rinv, polygon)
        self.pieces.append(Piece(orig))
        self.selected = len(self.pieces) - 1
        self._rebuild_rows()
        self._update_rows()
        self.render()
        return self.selected

    def edit_piece(self, i):
        """Open piece i in its own converter window, with full-size preview,
        rotation and its own settings. Returns the new window's app."""
        if not self._ready():
            return None
        flat, mask = core.piece_source(self.flat, self.mask, self.piece_poly(i))
        win = tk.Toplevel(self.root)
        editor = DoodleApp(win)
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        win.geometry(f"1350x800+{x + 40}+{max(y - 10, 0)}")
        editor.load_prepared(flat, mask, self.path, self._piece_name(i)[:-4],
                             self.settings(), f"piece {i + 1}")
        return editor

    def delete_piece(self, i):
        del self.pieces[i]
        if self.selected is not None:
            if self.selected == i:
                self.selected = min(i, len(self.pieces) - 1) if self.pieces else None
            elif self.selected > i:
                self.selected -= 1
        self._rebuild_rows()
        self._update_rows()
        self.render()

    def clear_pieces(self):
        self.pieces, self.selected = [], None
        self._rebuild_rows()

    def clear_dialog(self):
        if self.pieces and messagebox.askyesno("Clear pieces", "Remove all pieces?",
                                               parent=self.root):
            self.clear_pieces()
            self.render()

    def select_piece(self, i):
        self.selected = i
        for j, widgets in enumerate(self._rows):
            bg = ROW_SELECTED_BG if j == i else ROW_BG
            for widget in widgets:
                widget.configure(bg=bg)
        self.render()

    def save_piece(self, i, p):
        if not self._ready():
            return False
        poly, pad = self.piece_poly(i), self.pad.get()
        return self._write(p, lambda: core.to_rgba(core.cut_piece(self.alpha, poly, pad)))

    def save_piece_dialog(self, i):
        p = filedialog.asksaveasfilename(
            title=f"Save piece {i + 1}", defaultextension=".png",
            filetypes=[("PNG image", "*.png")], parent=self.root,
            initialdir=str(self.path.parent), initialfile=self._piece_name(i))
        if p:
            self.save_piece(i, p)

    def save_all(self, folder):
        """Save every piece into `folder`; returns the written paths."""
        folder = Path(folder)
        written = []
        for i in range(len(self.pieces)):
            p = folder / self._piece_name(i)
            if not self.save_piece(i, p):
                break
            written.append(p)
        if written:
            self.status.set(f"Saved {len(written)} piece(s) to {folder}")
        return written

    def save_all_dialog(self):
        if not self.pieces:
            messagebox.showinfo("No pieces", "Drag around a drawing on the preview first.",
                                parent=self.root)
            return
        folder = filedialog.askdirectory(title="Save all pieces to…",
                                         initialdir=str(self.path.parent), parent=self.root)
        if not folder:
            return
        existing = [n for n in (self._piece_name(i) for i in range(len(self.pieces)))
                    if (Path(folder) / n).exists()]
        if existing and not messagebox.askyesno(
                "Overwrite?", f"{len(existing)} file(s) already exist, e.g. {existing[0]}.\n"
                              "Overwrite them?", parent=self.root):
            return
        self.save_all(folder)

    def _rebuild_rows(self):
        for child in self.list_inner.winfo_children():
            child.destroy()
        self._rows, self._thumbs = [], []
        if not hasattr(self, "_blank"):  # sizes empty thumbnails in pixels, not characters
            self._blank = tk.PhotoImage(width=THUMB, height=THUMB)
        for i in range(len(self.pieces)):
            bg = ROW_SELECTED_BG if i == self.selected else ROW_BG
            row = tk.Frame(self.list_inner, bg=bg, padx=4, pady=4)
            row.pack(fill="x", pady=2)
            thumb = tk.Label(row, bg=bg, image=self._blank, width=THUMB, height=THUMB)
            thumb.pack(side="left")
            info = tk.Frame(row, bg=bg)
            info.pack(side="left", fill="both", expand=True, padx=(6, 0))
            text = tk.Label(info, text=f"Piece {i + 1}", bg=bg, anchor="w", justify="left")
            text.pack(anchor="w")
            btns = tk.Frame(info, bg=bg)
            btns.pack(anchor="w", pady=(2, 0))
            for label, cmd in (("Edit…", self.edit_piece), ("Save…", self.save_piece_dialog),
                               ("✕ Remove", self.delete_piece)):
                ttk.Button(btns, text=label, width=11,
                           command=lambda i=i, cmd=cmd: cmd(i)).pack(anchor="w", pady=(2, 0))
            for widget in (row, thumb, text, info, btns):
                widget.bind("<Button-1>", lambda e, i=i: self.select_piece(i))
            self._rows.append((row, thumb, text, info, btns))
            self._thumbs.append(None)

    def _update_rows(self):
        """Refresh piece thumbnails and sizes from the current result."""
        if self.alpha is None:
            return
        pad = self.pad.get()
        for i in range(min(len(self.pieces), len(self._rows))):
            cut = core.cut_piece(self.alpha, self.piece_poly(i), pad)
            h, w = cut.shape
            k = min(THUMB / w, THUMB / h)
            small = cv2.resize(cut, (max(int(w * k), 1), max(int(h * k), 1)),
                               interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
            img = ImageTk.PhotoImage(Image.fromarray(self._composite(small)))
            thumb, text = self._rows[i][1:3]
            thumb.configure(image=img, width=THUMB, height=THUMB)
            self._thumbs[i] = img
            text.configure(text=f"Piece {i + 1}\n{w}×{h} px")

    # ---------------------------------------------------------- coordinates

    def to_image(self, cx, cy):
        return self.vx + cx / self.scale, self.vy + cy / self.scale

    def to_canvas(self, ix, iy):
        return (ix - self.vx) * self.scale, (iy - self.vy) * self.scale

    # ----------------------------------------------------------------- zoom

    def _canvas_size(self):
        return max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)

    def _fit_scale(self):
        cw, ch = self._canvas_size()
        h, w = self.flat.shape
        return min((cw - 20) / w, (ch - 20) / h, 1.0)

    def _fit_view(self):
        self.scale = self._fit_scale()
        self._clamp_view()

    def _clamp_view(self):
        """Center the photo along an axis where it fits, else keep it on screen."""
        cw, ch = self._canvas_size()
        h, w = self.flat.shape
        vw, vh = cw / self.scale, ch / self.scale
        self.vx = (w - vw) / 2 if vw >= w else min(max(self.vx, 0), w - vw)
        self.vy = (h - vh) / 2 if vh >= h else min(max(self.vy, 0), h - vh)
        self.zoom_text.set(f"{self.scale * 100:.0f}%")

    def zoom_at(self, cx, cy, factor):
        """Zoom by `factor`, keeping the photo point under (cx, cy) in place."""
        if self.flat is None:
            return
        self.zoom_to(self.scale * factor, cx, cy)

    def zoom_to(self, scale, cx=None, cy=None):
        if self.flat is None:
            return
        cw, ch = self._canvas_size()
        cx = cw / 2 if cx is None else cx
        cy = ch / 2 if cy is None else cy
        ix, iy = self.to_image(cx, cy)
        fit = self._fit_scale()
        self.scale = min(max(scale, min(fit, 1.0)), MAX_ZOOM)
        self.fit = math.isclose(self.scale, fit)
        self.vx, self.vy = ix - cx / self.scale, iy - cy / self.scale
        self._clamp_view()
        self.render()

    def zoom_step(self, factor):
        self.zoom_at(*(s / 2 for s in self._canvas_size()), factor)

    def zoom_fit(self):
        if self.flat is None:
            return
        self.fit = True
        self._fit_view()
        self.render()

    def _pan_start(self, e):
        if self.flat is not None:
            self._pan = (e.x, e.y, self.vx, self.vy)
            self.canvas.config(cursor="fleur")

    def _pan_move(self, e):
        if not self._pan:
            return
        x, y, vx, vy = self._pan
        self.vx = vx - (e.x - x) / self.scale
        self.vy = vy - (e.y - y) / self.scale
        self._clamp_view()
        self.render()

    def _pan_end(self, _e):
        self._pan = None
        self.canvas.config(cursor="crosshair")

    # ---------------------------------------------------------------- lasso

    def _lasso_start(self, e):
        if self.flat is None:
            return
        self._lasso = [self.to_image(e.x, e.y)]
        self._lasso_last = (e.x, e.y)

    def _lasso_move(self, e):
        if not self._lasso:
            return
        lx, ly = self._lasso_last
        if abs(e.x - lx) + abs(e.y - ly) <= 2:
            return
        self._lasso.append(self.to_image(e.x, e.y))
        self._lasso_last = (e.x, e.y)
        self.canvas.create_line(lx, ly, e.x, e.y, fill=SELECTED_COLOR, width=2, tags="lasso")

    def _lasso_end(self, _e):
        pts, self._lasso = self._lasso, None
        self.canvas.delete("lasso")
        if not pts or len(pts) < 3:
            return
        pts = np.array(pts, np.float32)
        if cv2.contourArea(pts) * self.scale ** 2 < 64:  # a click or a scribble
            return
        h, w = self.flat.shape
        pts[:, 0] = pts[:, 0].clip(0, w - 1)
        pts[:, 1] = pts[:, 1].clip(0, h - 1)
        i = self.add_piece(pts.round().astype(np.int32))
        self.status.set(f"Added piece {i + 1}. Drag around another drawing, or save.")

    def _lasso_cancel(self, _=None):
        self._lasso = None
        self.canvas.delete("lasso")

    # -------------------------------------------------------------- preview

    def on_resize(self, _event):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(100, self._after_resize)

    def _after_resize(self):
        self._resize_job = None
        if self.flat is None:
            self._show_hint()
            return
        if self.fit:
            self._fit_view()
        else:
            self._clamp_view()
        self.render()

    def schedule_refresh(self, *_):
        if self._refresh_job:
            self.root.after_cancel(self._refresh_job)
        self._refresh_job = self.root.after(40, self.refresh)

    def refresh(self):
        """Recompute the full-size result for the current settings, then draw."""
        self._refresh_job = None
        if self.rflat is None:
            return
        self.alpha_full = core.finish_alpha(self.rflat, self.rmask, self.settings())
        if self.cropping and self.crop_fit.get():   # box follows the drawing
            self._set_edit_rect(self._ink_rect())
        self._set_work_rect(self._target_rect())
        self._update_rows()
        self.render()

    def _full_rect(self):
        h, w = self.rflat.shape
        return 0, 0, w, h

    def _ink_rect(self):
        """The drawing's bounding box plus padding (the whole image if empty)."""
        return core.crop_box(self.alpha_full, self.pad.get()) or self._full_rect()

    def _target_rect(self):
        """Which part of the rotated photo is the working image right now."""
        if self.cropping:
            return self._full_rect()
        if self.crop_fit_saved:
            return self._ink_rect()
        if self.crop_corners is not None:
            h, w = self.rflat.shape
            return core.rect_from_corners(self.Rrot, self.crop_corners, (w, h))
        return self._full_rect()

    def _set_work_rect(self, rect):
        old = self.work_rect
        x, y, w, h = rect
        self.flat = self.rflat[y:y + h, x:x + w]
        self.mask = None if self.rmask is None else self.rmask[y:y + h, x:x + w]
        self.alpha = self.alpha_full[y:y + h, x:x + w]
        shift = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], np.float64)
        self.R = shift @ self.Rrot
        self.Rinv = np.linalg.inv(self.R)
        self.work_rect = rect
        if self.fit:
            self._fit_view()
        else:
            if old is not None:      # keep the same content under the view
                self.vx += old[0] - x
                self.vy += old[1] - y
            self._clamp_view()

    # ------------------------------------------------------------------ crop

    def _corners(self, rect):
        """Rotated-photo rect -> its corners in original photo coords."""
        return core.transform_points(np.linalg.inv(self.Rrot), core.rect_corners(rect))

    def edit_rect(self):
        """The crop box being edited, (x, y, w, h) in rotated-photo coords."""
        h, w = self.rflat.shape
        return core.rect_from_corners(self.Rrot, self.edit_corners, (w, h))

    def start_crop(self):
        if self.rflat is None or self.cropping:
            return
        self._flush()
        self._lasso_cancel()
        self.cropping = True
        full = self._full_rect()
        if self.crop_fit_saved:
            self.crop_fit.set(True)
            self._fit_backup = self._corners(full)
            self.edit_corners = self._corners(self._ink_rect())
        else:
            self.crop_fit.set(False)
            self._fit_backup = None
            self.edit_corners = (self.crop_corners.copy() if self.crop_corners is not None
                                 else self._corners(full))
        self.tips.pack_forget()
        self.crop_section.pack(fill="x", pady=(6, 0))
        self.crop_button.state(["disabled"])
        self._update_spin_limits()
        self._sync_spins()
        self.fit = True
        self.status.set("Crop: drag the box's edges or type values, "
                        "then Save crop (Enter) or Cancel (Esc).")
        self.refresh()

    def save_crop(self):
        if not self.cropping:
            return
        rect, full = self.edit_rect(), self._full_rect()
        if self.crop_fit.get():
            self.crop_fit_saved, self.crop_corners = True, None
        elif rect == full:
            self.crop_fit_saved, self.crop_corners = False, None
        else:
            self.crop_fit_saved, self.crop_corners = False, self.edit_corners.copy()
        self._leave_crop_mode()
        self.refresh()
        if self.crop_fit_saved:
            self.status.set("Cropped to the drawing (follows the sliders).")
        elif self.crop_corners is None:
            self.status.set("No crop: using the whole image.")
        else:
            self.status.set(f"Cropped to {rect[2]}×{rect[3]} at ({rect[0]}, {rect[1]}).")

    def cancel_crop(self):
        if not self.cropping:
            return
        self._leave_crop_mode()
        self.refresh()
        self.status.set("Crop cancelled.")

    def _leave_crop_mode(self):
        if not self.cropping:
            return
        self.cropping = False
        self._crop_drag = None
        self._fit_backup = None
        self.crop_section.pack_forget()
        self.crop_holder.configure(height=1)   # Tk keeps an emptied frame's size otherwise
        self.crop_button.state(["!disabled"])
        if not self.tips.winfo_manager():
            self.tips.pack(anchor="w", pady=(10, 0))
        self.canvas.config(cursor="crosshair")
        self.fit = True

    def set_crop_box(self, x, y, w, h, user=True):
        """Move/resize the crop box (clamped to the image)."""
        self._set_edit_rect((x, y, w, h), user)
        self.render()

    def _set_edit_rect(self, rect, user=False):
        H, W = self.rflat.shape
        x, y, w, h = (int(round(v)) for v in rect)
        x, y = min(max(x, 0), W - 1), min(max(y, 0), H - 1)
        w, h = min(max(w, 1), W - x), min(max(h, 1), H - y)
        self.edit_corners = self._corners((x, y, w, h))
        if user and self.crop_fit.get():    # hand edits replace "to the drawing"
            self.crop_fit.set(False)
            self._fit_backup = None
        self._sync_spins()

    def on_crop_fit(self):
        if not self.cropping:
            return
        if self.crop_fit.get():
            self._fit_backup = self.edit_corners.copy()
            self._set_edit_rect(self._ink_rect())
        elif self._fit_backup is not None:
            self.edit_corners, self._fit_backup = self._fit_backup, None
            self._sync_spins()
        self.render()

    def _sync_spins(self):
        for key, v in zip("xywh", self.edit_rect()):
            self.crop_vars[key].set(str(v))

    def _update_spin_limits(self):
        H, W = self.rflat.shape
        limits = {"x": (0, W - 1), "y": (0, H - 1), "w": (1, W), "h": (1, H)}
        for key, (lo, hi) in limits.items():
            self.crop_spins[key].configure(from_=lo, to=hi)

    def _crop_from_spins(self):
        """Typed or arrow-clicked values -> crop box."""
        if not self.cropping:
            return
        try:
            values = tuple(int(float(self.crop_vars[k].get())) for k in "xywh")
        except ValueError:
            self._sync_spins()
            return
        if values != self.edit_rect():
            self.set_crop_box(*values)

    # Dragging the crop box

    _CROP_CURSORS = {"l": "sb_h_double_arrow", "r": "sb_h_double_arrow",
                     "t": "sb_v_double_arrow", "b": "sb_v_double_arrow",
                     "tl": "size_nw_se", "br": "size_nw_se",
                     "tr": "size_ne_sw", "bl": "size_ne_sw", "move": "fleur"}

    def _crop_hit(self, cx, cy, tol=8):
        """Which part of the crop box is at canvas point (cx, cy):
        'l'/'r'/'t'/'b' edges, corners like 'tl', 'move' inside, or None."""
        x, y, w, h = self.edit_rect()
        ax, ay = self.to_canvas(x, y)
        bx, by = self.to_canvas(x + w, y + h)
        horiz = vert = ""
        if ay - tol <= cy <= by + tol:
            dl, dr = abs(cx - ax), abs(cx - bx)
            if min(dl, dr) <= tol:
                horiz = "l" if dl <= dr else "r"
        if ax - tol <= cx <= bx + tol:
            dt, db = abs(cy - ay), abs(cy - by)
            if min(dt, db) <= tol:
                vert = "t" if dt <= db else "b"
        if horiz or vert:
            return vert + horiz
        if ax < cx < bx and ay < cy < by:
            return "move"
        return None

    def _crop_hover(self, e):
        if not self.cropping or self._crop_drag or self._pan:
            return
        cursor = self._CROP_CURSORS.get(self._crop_hit(e.x, e.y), "crosshair")
        try:
            self.canvas.config(cursor=cursor)
        except tk.TclError:          # cursor name not available on this system
            self.canvas.config(cursor="fleur")

    def _crop_press(self, e):
        mode = self._crop_hit(e.x, e.y)
        if mode:
            self._crop_drag = (mode, self.to_image(e.x, e.y), self.edit_rect())

    def _crop_move(self, e):
        if not self._crop_drag:
            return
        mode, (sx, sy), (x, y, w, h) = self._crop_drag
        px, py = self.to_image(e.x, e.y)
        H, W = self.rflat.shape
        x0, y0, x1, y1 = x, y, x + w, y + h     # x1/y1 are exclusive edges
        if mode == "move":
            dx = min(max(round(px - sx), -x0), W - x1)
            dy = min(max(round(py - sy), -y0), H - y1)
            x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
        else:
            if "l" in mode:
                x0 = min(max(round(px), 0), x1 - 1)
            if "r" in mode:
                x1 = min(max(round(px), x0 + 1), W)
            if "t" in mode:
                y0 = min(max(round(py), 0), y1 - 1)
            if "b" in mode:
                y1 = min(max(round(py), y0 + 1), H)
        self.set_crop_box(x0, y0, x1 - x0, y1 - y0)

    def _crop_release(self, e):
        self._crop_drag = None
        self._crop_hover(e)

    def _on_escape(self, _e=None):
        if self.cropping:
            self.cancel_crop()
        else:
            self._lasso_cancel()

    def _on_return(self, _e=None):
        if self.cropping:
            self.save_crop()

    def _background(self, w, h):
        key = (self.bg_mode.get(), w, h)
        if key not in self._bg_cache:
            if len(self._bg_cache) > 50:
                self._bg_cache.clear()
            mode = self.bg_mode.get()
            if mode == "Dark":
                bg = np.full((h, w, 3), 40, np.float32)
            elif mode == "Checker":
                yy, xx = np.indices((h, w))
                cells = ((yy // 12 + xx // 12) % 2).astype(np.float32)
                bg = np.repeat((150 + 60 * cells)[:, :, None], 3, axis=2)
            else:
                bg = np.full((h, w, 3), 245, np.float32)
            self._bg_cache[key] = bg
        return self._bg_cache[key]

    def _composite(self, alpha):
        """Alpha -> RGB preview: tint color over the preview background."""
        h, w = alpha.shape
        a = (alpha.astype(np.float32) / 255)[:, :, None]
        out = self._background(w, h) * (1 - a) + np.array(self.tint, np.float32) * a
        return out.astype(np.uint8)

    def _view_alpha(self):
        """The visible part of the result, resized to screen pixels.
        Returns (alpha, x0, y0) with (x0, y0) in photo coords, or None."""
        cw, ch = self._canvas_size()
        H, W = self.alpha.shape
        s = self.scale
        x0, y0 = max(math.floor(self.vx), 0), max(math.floor(self.vy), 0)
        x1 = min(math.ceil(self.vx + cw / s), W)
        y1 = min(math.ceil(self.vy + ch / s), H)
        if x1 <= x0 or y1 <= y0:
            return None
        # Resize the *result*, not the photo: shrinking the photo first would
        # blend thin lines into the paper until they fell below White.
        crop = self.alpha[y0:y1, x0:x1]
        tw, th = max(round((x1 - x0) * s), 1), max(round((y1 - y0) * s), 1)
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
        return cv2.resize(crop, (tw, th), interpolation=interp), x0, y0

    def render(self):
        """Draw the visible part of the result at the current zoom."""
        if self.alpha is None:
            return
        s = self.scale
        self.canvas.delete("all")
        view = self._view_alpha()
        if view is None:
            return
        shown, x0, y0 = view
        img = self._composite(shown)
        if self.cropping:   # dim everything outside the crop box
            bx, by, bw, bh = self.edit_rect()
            th, tw = img.shape[:2]
            ix0 = min(max(round((bx - x0) * s), 0), tw)
            ix1 = min(max(round((bx + bw - x0) * s), 0), tw)
            iy0 = min(max(round((by - y0) * s), 0), th)
            iy1 = min(max(round((by + bh - y0) * s), 0), th)
            dim = (img * 0.45).astype(np.uint8)
            dim[iy0:iy1, ix0:ix1] = img[iy0:iy1, ix0:ix1]
            img = dim
        self._photo = ImageTk.PhotoImage(Image.fromarray(img))
        px, py = self.to_canvas(x0, y0)
        self.canvas.create_image(round(px), round(py), anchor="nw", image=self._photo)

        for i in range(len(self.pieces)):
            pts = (self.piece_poly(i) - (self.vx, self.vy)) * s
            sel = i == self.selected
            color = SELECTED_COLOR if sel else PIECE_COLOR
            self.canvas.create_polygon(pts.ravel().tolist(), outline=color, fill="",
                                       width=3 if sel else 2)
            lx, ly = pts[:, 0].min(), pts[:, 1].min()
            self.canvas.create_text(lx + 4, ly + 4, text=str(i + 1), anchor="nw",
                                    fill=color, font=("Segoe UI", 12, "bold"))

        if self._lasso and len(self._lasso) > 1:  # loop in progress (e.g. zoomed mid-drag)
            pts = (np.array(self._lasso) - (self.vx, self.vy)) * s
            self.canvas.create_line(pts.ravel().tolist(), fill=SELECTED_COLOR,
                                    width=2, tags="lasso")

        if self.cropping:   # crop box outline and its 8 drag handles
            x, y, w, h = self.edit_rect()
            ax, ay = self.to_canvas(x, y)
            bx, by = self.to_canvas(x + w, y + h)
            self.canvas.create_rectangle(ax, ay, bx, by, outline=CROP_COLOR, width=2)
            mx, my = (ax + bx) / 2, (ay + by) / 2
            for hx, hy in ((ax, ay), (mx, ay), (bx, ay), (bx, my),
                           (bx, by), (mx, by), (ax, by), (ax, my)):
                self.canvas.create_rectangle(hx - 5, hy - 5, hx + 5, hy + 5,
                                             fill="white", outline=CROP_COLOR, width=2)


def main():
    try:  # crisp rendering on high-DPI Windows displays
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    app = DoodleApp(root)
    if len(sys.argv) > 1:  # e.g. a photo dropped onto the .exe
        root.after(100, lambda: app.open_path(sys.argv[1]))
    root.mainloop()


if __name__ == "__main__":
    main()
