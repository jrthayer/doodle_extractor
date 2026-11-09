"""Doodle Extractor: windowed app.

Open a photo of a drawing. "Extract ink" turns it into a Godot-ready PNG
(white lines, transparent everywhere else); without it the photo is kept as is.
Crop, rotate, and cut out parts of the photo as separate pieces.
Drag to move the image, scroll to zoom. "Lasso piece" switches the mouse to
drawing loops around drawings.
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

APP_NAME = "Doodle Extractor"
DEFAULT_WHITE = 255         # the app's White slider default (the CLI keeps 160)
HUG_WHITE = 160             # hugging finds ink as if White were at most this
IMAGE_TYPES = [("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
               ("All files", "*.*")]
BACKGROUNDS = ("Light", "Dark", "Checker")
THUMB = 96                    # piece thumbnail size (px)
MAX_ZOOM = 8.0                # 800%: screen px per photo px
ZOOM_STEP = 1.25
PIECE_COLOR = "#1e90ff"
SELECTED_COLOR = "#ff8c00"
CROP_COLOR = "#ff3b6b"
EDIT_TIPS = {"add": "Add area: loop more of the drawing, or click another piece",
             "cut": "Cut area: loop what to remove from this piece"}
BANNER_H = 34                 # mode banner height over the preview (px)
SLIDER_TRACK = 18             # slider track thickness (Tk's default is 15)
OPT_PADX, OPT_PADY = 8, 8     # padding inside the highlighted piece-settings slot
STEP_BTN_W = 22           # width of a slider's − / + button (px), taken from the slider
# Colors for the classic Tk widgets (ttk widgets get theirs from the ttk theme)
LIGHT = dict(bg="SystemButtonFace", fg="SystemButtonText", muted="#555555", canvas="#808080",
             row="#f4f4f4", row_sel="#dcdcdc", trough="SystemScrollbar",
             active="SystemButtonFace", field="SystemWindow", field_fg="SystemWindowText",
             button="#e4e4e4", hover="#d6d6d6", pressed="#c2c2c2", border="#adadad",
             disabled="#9a9a9a")
DARK = dict(bg="#202020", fg="#e8e8e8", muted="#a0a0a0", canvas="#2d2d2d",
            row="#2b2b2b", row_sel="#3d3d3d", trough="#2d2d2d", active="#3e3e3e",
            field="#2d2d2d", field_fg="#e8e8e8",
            button="#333333", hover="#3e3e3e", pressed="#555555", border="#4a4a4a",
            disabled="#777777")
# "pressed" is also the "on" color of toggle buttons and the highlight behind the
# piece settings that belong to them
_THEME = {"dark": None}       # None = not decided yet (follows Windows on first window)


def system_prefers_dark():
    """Windows' "app mode" setting (Settings > Personalization > Colors)."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows"
                            r"\CurrentVersion\Themes\Personalize") as key:
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except Exception:
        return False


def palette():
    return DARK if _THEME["dark"] else LIGHT


def _configure_style(root, dark):
    """Set the ttk theme for the whole program (all windows share it)."""
    style = ttk.Style(root)
    p = DARK if dark else LIGHT
    style.theme_use("clam")   # the native Windows theme can't be recolored
    style.configure(".", background=p["bg"], foreground=p["fg"],
                    fieldbackground=p["field"], bordercolor=p["border"],
                    lightcolor=p["button"], darkcolor=p["button"],
                    troughcolor=p["trough"], selectbackground=p["pressed"],
                    selectforeground=p["fg"], insertcolor=p["fg"],
                    arrowcolor=p["fg"], focuscolor=p["fg"])
    style.map(".", background=[("disabled", p["bg"])],
              foreground=[("disabled", p["disabled"])])
    for name in ("TButton", "TCombobox", "TSpinbox", "TScrollbar"):
        style.configure(name, background=p["button"], lightcolor=p["button"],
                        darkcolor=p["button"], bordercolor=p["border"],
                        arrowcolor=p["fg"])
    hover = [("pressed", p["pressed"]), ("active", p["hover"])]
    style.map("TButton", background=hover, lightcolor=hover, darkcolor=hover)
    style.map("TScrollbar", background=[("active", p["hover"])])
    style.configure("TCheckbutton", indicatorbackground=p["field"],
                    indicatorforeground=p["fg"])
    style.map("TCheckbutton", background=[("active", p["bg"])],
              indicatorbackground=[("pressed", p["hover"]), ("disabled", p["bg"])])
    style.map("TCombobox", fieldbackground=[("readonly", p["field"])],
              foreground=[("readonly", p["fg"])],
              selectbackground=[("readonly", p["field"])],
              selectforeground=[("readonly", p["fg"])],
              background=[("active", p["hover"])])
    style.map("TSpinbox", background=[("active", p["hover"])])
    style.configure("TLabelframe", background=p["bg"], bordercolor=p["border"])
    style.configure("TLabelframe.Label", background=p["bg"], foreground=p["fg"])
    style.configure("TSeparator", background=p["border"])
    # The piece settings slot, highlighted in the "on" color of its button
    style.configure("Opts.TFrame", background=p["pressed"])
    style.configure("Opts.TLabel", background=p["pressed"])
    style.configure("Opts.TCheckbutton", background=p["pressed"])
    style.map("Opts.TCheckbutton", background=[("active", p["pressed"])])
    list_bg, list_fg = p["field"], p["fg"]
    # Slider − / +: as short as the track, so keep only the button's frame and a
    # centered label (no focus ring or padding eating the room)
    frame, opts = style.layout("TButton")[0]
    style.layout("Step.TButton", [(frame, {**{k: v for k, v in opts.items() if k != "children"},
                                           "children": [("Button.label", {"sticky": ""})]})])
    style.configure("Step.TButton", padding=0, font=("Segoe UI", 9, "bold"))
    # Combobox drop-down lists made from now on (existing ones are fixed per window)
    root.option_add("*TCombobox*Listbox.background", list_bg)
    root.option_add("*TCombobox*Listbox.foreground", list_fg)


def theme_icon(master, kind, rgb, size=20):
    """A sun or moon icon as a Tk image, drawn smooth by supersampling."""
    from PIL import ImageDraw
    k = 4
    n = size * k
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = n / 2
    if kind == "sun":
        r = n * 0.2
        d.ellipse((c - r, c - r, c + r, c + r), fill=rgb)
        for i in range(8):
            a = i * math.pi / 4
            r0, r1 = n * 0.3, n * 0.45
            d.line((c + r0 * math.cos(a), c + r0 * math.sin(a),
                    c + r1 * math.cos(a), c + r1 * math.sin(a)), fill=rgb, width=int(n * 0.08))
    else:   # crescent: a disc with a smaller offset disc cut out
        r = n * 0.4
        d.ellipse((c - r, c - r, c + r, c + r), fill=rgb)
        cut = Image.new("L", (n, n), 0)
        ImageDraw.Draw(cut).ellipse((c - r * 0.35, c - r * 1.25, c + r * 1.45, c + r * 0.55),
                                    fill=255)
        alpha = np.array(img.getchannel("A"))
        alpha[np.array(cut) > 0] = 0
        img.putalpha(Image.fromarray(alpha))
    return ImageTk.PhotoImage(img.resize((size, size), Image.LANCZOS), master=master)


def set_theme(dark):
    """Switch every open window between the light and dark theme."""
    _THEME["dark"] = bool(dark)
    if _APPS:
        _configure_style(_APPS[0].root, _THEME["dark"])
    for app in list(_APPS):
        app.dark.set(_THEME["dark"])
        app._apply_theme()


@dataclass
class Piece:
    poly: np.ndarray      # outline in ORIGINAL photo coords (float), so it follows photo rotation
    loop: np.ndarray = None   # the loop as drawn (original coords), to re-hug it later
    hugged: bool = False      # poly was snapped to the drawing (smoothing applies)
    smooth: int = 0           # smoothing used for poly
    seed: np.ndarray = None   # Auto select: the clicked point (original coords)
    reach: int = 50           # Auto select: how far it grouped nearby strokes
    edited: bool = False      # areas were added or cut: poly is a list of contours and
                              # Smoothing / Reach leave it alone


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
        root.title(APP_NAME)
        root.geometry("1350x800")
        root.minsize(900, 500)

        self.path = None          # current photo
        self.base = None          # file-name stem for saving
        self.is_piece = False     # True in a piece-editor window
        self.bgr = None           # full-res photo to prepare from (None in piece editors)
        self.color0 = None        # color layer, unrotated (what's shown without Extract ink)
        self.valid0 = None        # its alpha (None = fully opaque)
        self.rcolor = self.rvalid = None   # ... after photo rotation
        self.color = self.valid = None     # ... cut to the working rect
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
        self.white = tk.IntVar(value=DEFAULT_WHITE)
        self.min_speck = tk.IntVar(value=0)
        self.pad = tk.IntVar(value=0)    # margin kept around drawings (no longer a slider)
        self.crop_fit = tk.BooleanVar(value=False)   # "Crop to the drawing" in crop mode
        self.select_mode = tk.BooleanVar(value=False)  # left-drag draws loops instead of panning
        self.extract = tk.BooleanVar(value=False)      # show/save extracted ink instead of the photo
        self.hug = tk.BooleanVar(value=True)           # snap new loops to the drawing's outline
        self.smooth = tk.IntVar(value=30)              # how much hugged outlines are rounded
        self.reach = tk.IntVar(value=50)               # Auto select: how much a click grabs
        self.pick_mode = tk.BooleanVar(value=False)    # left-click picks objects
        self.edit_mode = None     # (piece index, "add" | "cut"): loops change that piece
        self._opts_shown = None   # which piece settings the panel shows (see _show_opts)
        self._pick_press = None
        self._repick_job = None
        self._smooth_job = None
        self.crop_vars = {k: tk.StringVar(value="0") for k in ("x", "y", "w", "h")}
        self.detect_paper = tk.BooleanVar(value=True)
        self.bg_mode = tk.StringVar(value="Light")
        self.zoom_text = tk.StringVar(value="")
        self.angle = tk.DoubleVar(value=0.0)        # photo straighten, degrees clockwise
        if _THEME["dark"] is None:
            _THEME["dark"] = system_prefers_dark()
        self.dark = tk.BooleanVar(value=_THEME["dark"])
        self._scales, self._muted = [], []   # classic Tk sliders / gray hint labels to recolor
        self._opt_scales = []                # sliders on the highlighted piece settings
        _configure_style(root, _THEME["dark"])

        self._build_ui()
        self.set_extract(False, redraw=False)   # default: the photo as is
        self._apply_theme()
        root.bind("<Map>", lambda e: e.widget is root and self._title_bar(), add="+")
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
            for name in ("_resize_job", "_refresh_job", "_rotate_job", "_repick_job",
                         "_smooth_job", "_step_job"):   # timers would fire on a dead window
                job = getattr(self, name, None)
                if job:
                    try:
                        self.root.after_cancel(job)
                    except tk.TclError:
                        pass

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side="top", fill="x")
        tip = self._tip
        tip(ttk.Button(bar, text="Open…", command=self.open_dialog),
            "Open a photo (Ctrl+O)").pack(side="left")
        tip(ttk.Button(bar, text="Save PNG…", command=self.save_dialog),
            "Save the image, cropped and rotated, as a PNG (Ctrl+S)").pack(side="left",
                                                                           padx=(6, 18))

        ttk.Label(bar, text="Preview background:").pack(side="left")
        bg = self.bg_combo = ttk.Combobox(bar, textvariable=self.bg_mode, values=BACKGROUNDS,
                                          state="readonly", width=8)
        tip(bg, "Background behind the preview only; saved PNGs stay transparent")
        bg.pack(side="left", padx=(4, 12))
        bg.bind("<<ComboboxSelected>>", lambda e: self.schedule_refresh())

        ttk.Label(bar, text="Preview line color:").pack(side="left")
        self.tint_swatch = tk.Button(bar, width=3, relief="solid", bd=1,
                                     bg=self._hex(self.tint), command=self.pick_tint)
        tip(self.tint_swatch, "Color of the lines in the preview only; saved ink is always white")
        self.tint_swatch.pack(side="left", padx=(4, 18))

        ttk.Label(bar, text="Zoom:").pack(side="left")
        tip(ttk.Button(bar, text="−", width=3, command=lambda: self.zoom_step(1 / ZOOM_STEP)),
            "Zoom out (Ctrl+minus)").pack(side="left", padx=(4, 0))
        ttk.Label(bar, textvariable=self.zoom_text, width=6, anchor="center").pack(side="left")
        tip(ttk.Button(bar, text="+", width=3, command=lambda: self.zoom_step(ZOOM_STEP)),
            "Zoom in (Ctrl+plus)").pack(side="left")
        tip(ttk.Button(bar, text="Fit", width=4, command=self.zoom_fit),
            "Fit the whole image in the window (Ctrl+0)").pack(side="left", padx=4)
        tip(ttk.Button(bar, text="100%", width=5, command=lambda: self.zoom_to(1.0)),
            "Show actual pixels").pack(side="left")
        # Theme switch: a moon in light mode (click for dark), a sun in dark mode
        self.theme_button = ttk.Label(bar, cursor="hand2")   # just the icon, clickable
        self.theme_button.pack(side="right", padx=(0, 4))
        self.theme_button.bind("<Button-1>", lambda e: self.toggle_theme())
        tip(self.theme_button, lambda: self.theme_tip)

        self.status = tk.StringVar(value="Open a photo to start (Ctrl+O).")
        ttk.Label(self.root, textvariable=self.status, anchor="w",
                  padding=(8, 3), relief="sunken").pack(side="bottom", fill="x")

        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True)

        self._build_settings(body)
        self._build_pieces_panel(body)

        self.canvas = tk.Canvas(body, bg="#808080", highlightthickness=0, cursor="fleur")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_resize)
        # Left button: crop box in crop mode, lasso in select mode, else pan
        self.canvas.bind("<ButtonPress-1>", self._left_press)
        self.canvas.bind("<B1-Motion>", self._left_move)
        self.canvas.bind("<ButtonRelease-1>", self._left_release)
        self.canvas.bind("<Motion>", self._crop_hover)
        for b in (2, 3):  # middle or right drag pans
            self.canvas.bind(f"<ButtonPress-{b}>", self._pan_start)
            self.canvas.bind(f"<B{b}-Motion>", self._pan_move)
            self.canvas.bind(f"<ButtonRelease-{b}>", self._pan_end)
        self._show_hint()

    def _build_settings(self, parent):
        side = ttk.Frame(parent, padding=(10, 6))
        side.pack(side="left", fill="y")
        tip = self._tip
        # Extract ink / Crop / Rotate on one row; each opens its settings below
        bar = ttk.Frame(side)
        bar.pack(fill="x")
        self.extract_button = ttk.Button(bar, text="Extract ink", width=-6,
                                         command=self.toggle_extract_panel)
        tip(self.extract_button, "Show the ink extraction settings; click again to hide them")
        self.crop_button = ttk.Button(bar, text="Crop", width=-6, command=self.toggle_crop)
        tip(self.crop_button, "Show the crop box; click again to close it")
        self.rotate_button = ttk.Button(bar, text="Rotate", width=-6, command=self.toggle_rotate)
        tip(self.rotate_button, "Show the rotate and straighten settings; click again to hide "
                                "them")
        for n, b in enumerate((self.extract_button, self.crop_button, self.rotate_button)):
            b.pack(side="left", expand=True, fill="x", padx=(0 if n == 0 else 4, 0))

        self.extract_holder = ttk.Frame(side)   # keeps each section's place in the column
        self.extract_holder.pack(fill="x")
        self._build_extract_section(self.extract_holder)
        self.crop_holder = ttk.Frame(side)
        self.crop_holder.pack(fill="x")
        self._build_crop_section(self.crop_holder)
        self.rotate_holder = ttk.Frame(side)
        self.rotate_holder.pack(fill="x")
        self._build_rotate_section(self.rotate_holder)
        # Keep the column as wide as its widest section even when all are closed,
        # so the preview doesn't jump sideways when one opens
        side.update_idletasks()
        widest = max(sec.winfo_reqwidth() for sec in
                     (self.extract_section, self.crop_section, self.rotate_section))
        ttk.Frame(side, width=widest, height=1).pack()

    def _build_rotate_section(self, parent):
        """Rotation settings, shown only while the Rotate button is on. Like the
        other sections it's highlighted in the button's "on" color."""
        tip = self._tip
        sec = self.rotate_section = ttk.Frame(parent, style="Opts.TFrame",
                                              padding=(10, 8, 10, 10))
        row = ttk.Frame(sec, style="Opts.TFrame")
        row.pack(anchor="w")
        tip(ttk.Button(row, text="⟲ 90°", width=-6, command=lambda: self.rotate_photo(-1)),
            "Turn the photo a quarter turn left").pack(side="left")
        tip(ttk.Button(row, text="⟳ 90°", width=-6, command=lambda: self.rotate_photo(1)),
            "Turn the photo a quarter turn right").pack(side="left", padx=4)
        tip(ttk.Button(row, text="Reset", width=-6, command=self.reset_rotation),
            "Undo all rotation and straightening").pack(side="left")
        self.straighten_scale = self._slider(
            sec, "Straighten (degrees clockwise)", self.angle, -45, 45,
            lambda v: self.schedule_rotate(), resolution=0.5, length=220, style="Opts",
            tip="Rotate the photo slightly to fix a tilt; exposed corners become transparent")

    def toggle_rotate(self):
        self.set_rotate_open(not self.rotate_section.winfo_manager())

    def set_rotate_open(self, on):
        """Show or hide the rotate settings (the rotation itself stays)."""
        if on:
            self._close_other_sections("rotate")
            self.rotate_section.pack(fill="x", pady=(6, 0))
        else:
            self.rotate_section.pack_forget()
            self.rotate_holder.configure(height=1)   # Tk keeps an emptied frame's size
        self.rotate_button.state(["pressed"] if on else ["!pressed"])

    def _build_extract_section(self, parent):
        """Ink extraction settings, shown while the Extract ink button is on.
        Apply extraction turns it on; the rest only shows while it's ticked."""
        # Highlighted in the "on" color of the Extract ink button, like the piece settings
        sec = self.extract_section = ttk.Frame(parent, style="Opts.TFrame",
                                               padding=(10, 0, 10, 10))
        self.apply_check = self._tip(
            ttk.Checkbutton(sec, text="Apply extraction", variable=self.extract,
                            command=lambda: self.set_extract(self.extract.get()),
                            style="Opts.TCheckbutton"),
            "Turn the drawing into white lines on transparent, ready for Godot")
        self.apply_check.pack(anchor="w", pady=(10, 0))
        body = self.extract_body = ttk.Frame(sec, style="Opts.TFrame")
        self.black_scale = self._slider(
            body, "Black (solid ink at or below)", self.black, 0, 255, self.on_black, length=220,
            style="Opts", tip="Ink this dark or darker is fully solid. Raise it if line middles look gray.")
        self.white_scale = self._slider(
            body, "White (transparent at or above)", self.white, 0, 255, self.on_white, length=220,
            style="Opts", tip="Anything this light or lighter turns transparent. Lower it if the grid shows; "
                "raise it if lines break.")
        self.specks_scale = self._slider(
            body, "Remove specks smaller than (px)", self.min_speck, 0, 100,
            lambda v: self.schedule_refresh(), length=220, style="Opts",
            tip="Erase dots smaller than this many pixels")
        tip = ("• Grid showing? Lower White.\n"
               "• Lines broken? Raise White.\n"
               "• Line middles gray? Raise Black.\n"
               "• Color it in Godot: Self Modulate.")
        self.tips = ttk.Label(body, text=tip, justify="left", style="Opts.TLabel")
        self._muted.append(self.tips)
        self.tips.pack(anchor="w", pady=(10, 0))
        self.paper_check = ttk.Checkbutton(body, text="Detect paper (ignore background)",
                                           variable=self.detect_paper, command=self.reprepare,
                                           style="Opts.TCheckbutton")
        self._tip(self.paper_check, "Ignore the background around the page")
        self.paper_check.pack(anchor="w", pady=(10, 0))
        self._tip(ttk.Button(body, text="Reset sliders", command=self.reset),
                  "Put Black, White and Remove specks back to their defaults").pack(
            anchor="w", pady=(8, 0))

    def _build_crop_section(self, parent):
        """The crop settings, shown only while cropping."""
        # Highlighted in the "on" color of the Crop button, like the piece settings
        sec = self.crop_section = ttk.Frame(parent, style="Opts.TFrame", padding=10)
        self._tip(ttk.Checkbutton(sec, text="Crop to the drawing", variable=self.crop_fit,
                                  command=self.on_crop_fit, style="Opts.TCheckbutton"),
                  "Snap the box tight to the ink and follow the sliders").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        self.crop_spins = {}
        for i, (key, label) in enumerate((("x", "X"), ("y", "Y"), ("w", "Width"), ("h", "Height"))):
            r, c = divmod(i, 2)
            ttk.Label(sec, text=label, style="Opts.TLabel").grid(row=1 + r, column=2 * c, sticky="w",
                                            padx=(0 if c == 0 else 10, 4), pady=2)
            sp = ttk.Spinbox(sec, from_=0, to=99999, increment=1, width=7,
                             textvariable=self.crop_vars[key], command=self._crop_from_spins)
            sp.grid(row=1 + r, column=2 * c + 1, sticky="w", pady=2)
            sp.bind("<Return>", lambda e: self._crop_from_spins() or "break")
            sp.bind("<FocusOut>", lambda e: self._crop_from_spins())
            self.crop_spins[key] = sp
        btns = ttk.Frame(sec, style="Opts.TFrame")
        btns.grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))
        self._tip(ttk.Button(btns, text="Save crop", command=self.save_crop),
                  "Apply the crop (Enter)").pack(side="left")
        self._tip(ttk.Button(btns, text="Reset", command=self.reset_crop),
                  "Put the box back at the image edges").pack(side="left", padx=6)

    def _build_pieces_panel(self, parent):
        panel = ttk.Frame(parent, padding=(6, 6))
        panel.pack(side="right", fill="y")
        ttk.Label(panel, text="Pieces", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        hint = ttk.Label(panel, text="Auto select: click a drawing.\nLasso: drag around "
                                     "it.\nEsc stops, Delete removes.", justify="left")
        hint.pack(anchor="w")
        self._muted.append(hint)

        tip = self._tip
        btns = ttk.Frame(panel)
        btns.pack(side="bottom", fill="x", pady=(6, 0))
        tip(ttk.Button(btns, text="Save all…", command=self.save_all_dialog),
            "Save every piece to a folder").pack(side="left")
        tip(ttk.Button(btns, text="Clear", command=self.clear_dialog),
            "Remove all pieces").pack(side="left", padx=6)
        self.select_button = ttk.Button(panel, text="Lasso piece",
                                        command=lambda: self.set_select_mode(
                                            not self.select_mode.get()))
        tip(self.select_button, "Drag a loop around a drawing to make it a piece")
        self.select_button.pack(side="bottom", fill="x", pady=(6, 0))   # just above the row
        self.pick_button = ttk.Button(panel, text="Auto select piece",
                                      command=lambda: self.set_pick_mode(not self.pick_mode.get()))
        tip(self.pick_button, "Click a drawing to make it a piece; dragging still moves the image")
        self.pick_button.pack(side="bottom", fill="x", pady=(6, 0))

        # Piece settings slot: always the same height (so the list doesn't jump);
        # only the settings for the current mode show in it (see _show_opts)
        self.opts_frame = ttk.Frame(panel)
        self.opts_frame.pack(side="bottom", fill="x", pady=(8, 0))
        self.opts_frame.pack_propagate(False)
        frames = self._opt_frames = {k: ttk.Frame(self.opts_frame, style="Opts.TFrame")
                                     for k in ("hug", "reach", "smooth")}
        self.hug_check = tip(ttk.Checkbutton(frames["hug"], text="Hug the drawing",
                                             variable=self.hug, style="Opts.TCheckbutton"),
                             "Pull each new loop in to the drawing's outline")
        self.hug_check.pack(anchor="w", pady=(8, 0))
        self.reach_scale = self._slider(
            frames["reach"], "Reach (Auto select)", self.reach, 0, 100, self.schedule_repick,
            length=230, style="Opts",
            tip="How far Auto select groups nearby strokes into one piece "
                "(solid shapes: above 50)")
        self.smooth_scale = self._slider(
            frames["smooth"], "Smoothing", self.smooth, 0, 100, self.schedule_resmooth,
            length=230, style="Opts",
            tip="How much hugged outlines are rounded out; it never cuts off ink")
        panel.update_idletasks()
        h = {k: f.winfo_reqheight() for k, f in frames.items()}
        self.opts_frame.configure(height=max(h["hug"], h["reach"]) + h["smooth"]
                                  + OPT_PADY)

        wrap = ttk.Frame(panel)
        wrap.pack(fill="both", expand=True, pady=(6, 0))
        self.list_canvas = tk.Canvas(wrap, width=250, highlightthickness=0)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.list_canvas.yview)
        self.list_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.list_canvas.pack(side="left", fill="both", expand=True)
        self.list_inner = tk.Frame(self.list_canvas)
        self.list_canvas.create_window((0, 0), window=self.list_inner, anchor="nw")
        self.list_inner.bind("<Configure>", lambda e: self._update_list_region())
        self.list_canvas.bind("<Configure>", lambda e: self._update_list_region())

    def _list_overflows(self):
        """True when the piece cards are taller than the visible list."""
        box = self.list_canvas.bbox("all")
        return box is not None and box[3] > self.list_canvas.winfo_height()

    def _update_list_region(self):
        """Scroll region = the cards, but never shorter than the list itself:
        when everything fits there's nothing to scroll."""
        box = self.list_canvas.bbox("all") or (0, 0, 0, 0)
        h = max(box[3], self.list_canvas.winfo_height())
        self.list_canvas.configure(scrollregion=(0, 0, max(box[2], 1), h))
        if not self._list_overflows():
            self.list_canvas.yview_moveto(0)

    def _wheel(self, e):
        """Mouse wheel: zoom over the preview, scroll over the pieces list.
        Returns True if the pointer was over this window's widgets."""
        try:
            w = self.root.winfo_containing(e.x_root, e.y_root)
        except (tk.TclError, KeyError):
            return False
        if w is self.canvas:
            self.zoom_at(e.x_root - self.canvas.winfo_rootx(),
                         e.y_root - self.canvas.winfo_rooty(),
                         ZOOM_STEP if e.delta > 0 else 1 / ZOOM_STEP)
            return True
        while w is not None:
            if w is self.list_canvas:
                if self._list_overflows():
                    self.list_canvas.yview_scroll(int(-e.delta / 120), "units")
                return True
            w = w.master
        return False

    @staticmethod
    def _step_var(var, lo, hi, resolution, command, sign):
        """Move a slider variable one notch; returns False at the end of its range."""
        return DoodleApp._set_var(var, lo, hi, resolution, command,
                                  var.get() + sign * resolution)

    @staticmethod
    def _set_var(var, lo, hi, resolution, command, value):
        """Set a slider variable (snapped and clamped) and run its command, as
        dragging the slider would. Returns False if the value didn't change."""
        old = var.get()
        v = min(max(round(value / resolution) * resolution, lo), hi)
        if isinstance(var, tk.IntVar):
            v = int(v)
        if v == old:
            return False
        var.set(v)
        command(str(v))
        return True

    def _hold_repeat(self, step):
        """Run `step()` now and, while the button is held, again after 400 ms
        and then every 60 ms, until it returns False or `_stop_repeat()`."""
        self._stop_repeat()
        step()

        def repeat():
            self._step_job = self.root.after(60, repeat) if step() else None
        self._step_job = self.root.after(400, repeat)

    def _stop_repeat(self):
        if getattr(self, "_step_job", None):
            self.root.after_cancel(self._step_job)
        self._step_job = None

    def _slider(self, parent, label, var, lo, hi, command, resolution=1, length=240, tip=None,
                style=None):
        """A labelled slider with − / + buttons that step it one notch (hold to
        repeat). `tip` is its status-bar hint. Returns the tk.Scale; its row
        frame is `scale.master` and its step function `scale.step(+1 | -1)`.
        `style="Opts"` puts it on the highlighted piece-settings background."""
        frame_style = f"{style}.TFrame" if style else "TFrame"
        name = ttk.Label(parent, text=label, style=f"{style}.TLabel" if style else "TLabel")
        name.pack(anchor="w", pady=(8, 0))
        row = ttk.Frame(parent, style=frame_style)
        row.pack(anchor="w")
        length = max(length - 2 * STEP_BTN_W, 80)   # the buttons take the room
        scale = tk.Scale(row, from_=lo, to=hi, orient="horizontal", variable=var,
                         length=length, command=command, resolution=resolution,
                         width=SLIDER_TRACK)

        def step(sign):
            return self._step_var(var, lo, hi, resolution, command, sign)

        def press(sign):
            self._hold_repeat(lambda: step(sign))

        def release(_e=None):
            self._stop_repeat()

        # Buttons exactly as tall as the slider's track, bottoms lined up with it
        track = int(scale.cget("width")) + 2 * int(scale.cget("borderwidth"))
        edge = int(scale.cget("highlightthickness"))
        buttons, holders = [], []
        for sign, text in ((-1, "−"), (1, "+")):
            holder = ttk.Frame(row, width=STEP_BTN_W, height=track, style=frame_style)
            holder.pack_propagate(False)
            b = ttk.Button(holder, text=text, style="Step.TButton")
            b.pack(fill="both", expand=True)
            b.bind("<ButtonPress-1>", lambda e, s=sign: press(s))
            b.bind("<ButtonRelease-1>", release)
            b.bind("<Leave>", release, add="+")
            buttons.append(b)
            holders.append(holder)
        holders[0].pack(side="left", anchor="s", pady=(0, edge))
        scale.pack(side="left")
        holders[1].pack(side="left", anchor="s", pady=(0, edge))
        scale.step, scale.step_buttons = step, buttons
        if tip:
            for w in (name, scale, *buttons):
                self._tip(w, tip)
        (self._opt_scales if style == "Opts" else self._scales).append(scale)
        return scale

    def _apply_theme(self):
        """Recolor this window's classic Tk widgets for the current theme."""
        p = palette()
        self.root.configure(bg=p["bg"])
        self.canvas.configure(bg=p["canvas"])
        self.list_canvas.configure(bg=p["bg"])
        self.list_inner.configure(bg=p["bg"])
        self.tint_swatch.configure(highlightbackground=p["bg"])
        for scale in self._scales:
            scale.configure(bg=p["bg"], fg=p["fg"], troughcolor=p["trough"],
                            activebackground=p["active"], highlightbackground=p["bg"])
        for scale in self._opt_scales:      # on the highlighted piece settings
            scale.configure(bg=p["pressed"], fg=p["fg"], troughcolor=p["trough"],
                            activebackground=p["hover"], highlightbackground=p["pressed"])
        for label in self._muted:
            label.configure(foreground=p["muted"])
        self._color_rows()
        dark = _THEME["dark"]
        self._theme_icon = theme_icon(self.root, "sun" if dark else "moon",
                                      (230, 230, 230) if dark else (60, 60, 60))
        self.theme_button.configure(image=self._theme_icon)
        self.theme_tip = "Switch to the light theme" if dark else "Switch to the dark theme"
        try:   # a drop-down list that was already opened keeps its old colors otherwise
            pop = self.root.tk.call("ttk::combobox::PopdownWindow", self.bg_combo)
            self.root.tk.call(f"{pop}.f.l", "configure", "-background", p["field"],
                              "-foreground", p["field_fg"])
        except tk.TclError:
            pass
        self._title_bar()

    def toggle_theme(self):
        set_theme(not self.dark.get())

    def _hover_tip(self, text):
        """Show a button's hint in the status bar while the pointer is on it.
        On leaving, the old message comes back unless something (e.g. clicking
        the button) has put a new one there meanwhile."""
        if text:
            if getattr(self, "_tip_shown", None) is None:
                self._status_before = self.status.get()
            self._tip_shown = text
            self.status.set(text)
        elif getattr(self, "_tip_shown", None) is not None:
            if self.status.get() == self._tip_shown:
                self.status.set(self._status_before)
            self._tip_shown = self._status_before = None

    def _tip(self, widget, text):
        """Give `widget` a status-bar hint on hover. `text` may be a function
        (for hints that change, like the theme icon's). Returns the widget."""
        widget.bind("<Enter>", lambda e: self._hover_tip(text() if callable(text) else text),
                    add="+")
        widget.bind("<Leave>", lambda e: self._hover_tip(None), add="+")
        return widget

    def _title_bar(self):
        """Dark or light window title bar (Windows 10 20H1+ / 11)."""
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            if not hwnd:
                return
            on = ctypes.c_int(1 if _THEME["dark"] else 0)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(on),
                                                       ctypes.sizeof(on))
            # SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED: redraw the frame now
            ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x27)
        except Exception:
            pass

    def _color_rows(self):
        p = palette()
        for j, widgets in enumerate(self._rows):
            bg = p["row_sel"] if j == self.selected else p["row"]
            for widget in widgets:
                widget.configure(bg=bg)
            widgets[2].configure(fg=p["fg"])   # the "Piece N" label

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
        self.white.set(DEFAULT_WHITE)
        self.min_speck.set(d.min_speck)
        if not self.detect_paper.get() and self.bgr is not None:
            self.detect_paper.set(True)
            self.reprepare()
        self.schedule_refresh()

    def set_extract(self, on, redraw=True):
        """Apply extraction on: show/save the extracted ink (and its settings).
        Off: show/save the photo itself (the slider values are kept)."""
        self.extract.set(on)
        self.extract_button.configure(text="Extract ink ✓" if on else "Extract ink")
        self.tint_swatch.configure(state="normal" if on else "disabled")
        if on:
            self.extract_body.pack(fill="x", after=self.apply_check)
        else:
            self.extract_body.pack_forget()
        if redraw and self.rflat is not None:
            self.refresh()
            self.status.set("Extraction on: tune the sliders. Untick Apply extraction to show "
                            "the photo." if on else "Showing the photo as is.")

    def toggle_extract_panel(self):
        self.set_extract_open(not self.extract_section.winfo_manager())

    def _close_other_sections(self, keep):
        """Extract ink, Crop and Rotate: only one has its settings open at a time.
        Closing Crop this way keeps the crop box, like clicking Crop again."""
        if keep != "extract" and self.extract_section.winfo_manager():
            self.set_extract_open(False)
        if keep != "rotate" and self.rotate_section.winfo_manager():
            self.set_rotate_open(False)
        if keep != "crop" and self.cropping:
            self.save_crop()

    def set_extract_open(self, on):
        """Show or hide the Extract ink settings (extraction itself is unchanged)."""
        if on:
            self._close_other_sections("extract")
            self.extract_section.pack(fill="x", pady=(6, 0))
        else:
            self.extract_section.pack_forget()
            self.extract_holder.configure(height=1)   # Tk keeps an emptied frame's size
        self.extract_button.state(["pressed"] if on else ["!pressed"])

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
        self.color0, self.valid0 = self.bgr, None
        self.root.title(f"{APP_NAME}: {p.name}")
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

    def load_prepared(self, flat, mask, path, base, settings, label,
                      color=None, valid=None, extract=True):
        """Load an already-prepared image (a piece from another window).
        `color`/`valid` are the matching unfiltered photo and its alpha."""
        self.bgr = None                     # nothing to re-prepare from
        self.path, self.base, self.is_piece = Path(path), base, True
        if color is None:                   # no photo given: show the prepared gray
            color = cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)
        self.color0, self.valid0 = color, valid
        self.root.title(f"{APP_NAME}: {self.path.name}, {label}")
        self.paper_check.state(["disabled"])
        self._reset_for_new_image(fit_crop=True)   # start cropped to the drawing
        self.black.set(settings.black)
        self.white.set(settings.white)
        self.min_speck.set(settings.min_speck)
        self.pad.set(settings.pad)
        self.flat0, self.mask0 = flat, mask
        self.set_extract(extract, redraw=False)
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
                        "Drag to move, scroll to zoom.")
        self.apply_rotation()

    # ------------------------------------------------------- photo rotation

    def apply_rotation(self, refit=False):
        """Rotate the prepared photo by the current rotation and redraw."""
        self._rotate_job = None
        if self.flat0 is None:
            return
        angle = float(self.angle.get())
        self.rflat, self.rmask = core.rotate_prepared(self.flat0, self.mask0, self.quarter, angle)
        self.rcolor, self.rvalid = core.rotate_color(self.color0, self.valid0, self.quarter, angle)
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
            messagebox.showinfo("Finish cropping", "Click Save crop (or Crop) first.",
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
        return self._write(p, self._image_rgba)

    # What gets saved: extracted ink (white + alpha) or the photo itself (BGRA)

    def _image_rgba(self):
        if self.extract.get():
            return core.to_rgba(self.alpha)
        return core.color_rgba(self.color, self.valid)

    def _piece_rgba(self, i):
        poly = self.piece_poly(i)
        if self.extract.get():
            return core.to_rgba(core.cut_piece(self.alpha, poly, self.pad.get()))
        return core.color_piece(self.color, self.valid, poly)

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
        """Piece i's loop in current (rotated) photo coords. For an edited piece,
        a list of contours (outlines and holes)."""
        poly = self.pieces[i].poly
        if isinstance(poly, list):
            return [core.transform_points(self.R, c).round().astype(np.int32) for c in poly]
        return core.transform_points(self.R, poly).round().astype(np.int32)

    def add_piece(self, polygon, loop=None, hugged=False, smooth=0):
        """Add a loop given in current (rotated) photo coords. `loop` is the loop
        as drawn when `polygon` is its hugged outline."""
        orig = core.transform_points(self.Rinv, polygon)
        raw = None if loop is None else core.transform_points(self.Rinv, loop)
        self.pieces.append(Piece(orig, raw, hugged, smooth))
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
        poly = self.piece_poly(i)
        flat, mask = core.piece_source(self.flat, self.mask, poly)
        color, valid = core.color_source(self.color, self.valid, poly)   # same box
        win = tk.Toplevel(self.root)
        editor = DoodleApp(win)
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        win.geometry(f"1350x800+{x + 40}+{max(y - 10, 0)}")
        editor.load_prepared(flat, mask, self.path, self._piece_name(i)[:-4],
                             self.settings(), f"piece {i + 1}",
                             color=color, valid=valid, extract=self.extract.get())
        return editor

    def delete_piece(self, i):
        if self.edit_mode:
            j, op = self.edit_mode
            if j == i:
                self.set_edit_mode(None)
            elif j > i:
                self.edit_mode = (j - 1, op)
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
        if self.edit_mode:
            self.set_edit_mode(None)
        self.pieces, self.selected = [], None
        self._rebuild_rows()

    def clear_dialog(self):
        if self.pieces and messagebox.askyesno("Clear pieces", "Remove all pieces?",
                                               parent=self.root):
            self.clear_pieces()
            self.render()

    def select_piece(self, i):
        self.selected = i
        if self.pieces[i].hugged:     # show its smoothing (setting the var doesn't re-smooth)
            self.smooth.set(self.pieces[i].smooth)
        if self.pieces[i].seed is not None:
            self.reach.set(self.pieces[i].reach)
        self._color_rows()
        self.render()

    def _hug_alpha(self):
        """The ink that hugging looks for. A high White (e.g. 255) brings back the
        faint grid, which would count as drawing, so hugging caps White."""
        s = self.settings()
        if s.white <= HUG_WHITE:
            return self.alpha
        return core.make_alpha(self.flat, self.mask, s.black, max(HUG_WHITE, s.black + 1))

    def schedule_resmooth(self, *_):
        if self._smooth_job:
            self.root.after_cancel(self._smooth_job)
        self._smooth_job = self.root.after(80, self.resmooth)

    def resmooth(self):
        """Re-hug the selected piece with the current Smoothing value."""
        self._smooth_job = None
        i = self.selected
        if i is None or i >= len(self.pieces) or self.alpha is None:
            return
        piece, v = self.pieces[i], self.smooth.get()
        if piece.edited:
            return
        if piece.seed is not None:
            self.repick(smooth_only=True)
            return
        if not piece.hugged or piece.loop is None:
            return
        loop = core.transform_points(self.R, piece.loop).round().astype(np.int32)
        hugged = core.hug_loop(self._hug_alpha(), loop, smooth=v, color=self.color)
        if hugged is None:
            return
        piece.poly, piece.smooth = core.transform_points(self.Rinv, hugged), v
        self._update_rows()
        self.render()
        self.status.set(f"Piece {i + 1} smoothing: {v}")

    # ---------------------------------------------------------- auto select

    def _pick(self, seed, reach, smooth):
        """Outline of the object at `seed` (working coords), or None."""
        return core.pick_object(self._hug_alpha(), self.color, seed, reach, smooth)

    def pick_at(self, x, y):
        """Pick the object at (x, y) in working coords and add it as a piece."""
        if self.alpha is None:
            return None
        self._flush()
        reach, smooth = self.reach.get(), self.smooth.get()
        poly = self._pick((x, y), reach, smooth)
        if poly is None:
            self.status.set("Nothing to pick there; click on a line or shape.")
            return None
        i = self.add_piece(poly, hugged=True, smooth=smooth)
        piece = self.pieces[i]
        piece.seed = core.transform_points(self.Rinv, np.array([[x, y]], float))[0]
        piece.reach = reach
        self.status.set(f"Added piece {i + 1} (picked by click). Reach adjusts how much it "
                        "grabs.")
        return i

    def schedule_repick(self, *_):
        if self._repick_job:
            self.root.after_cancel(self._repick_job)
        self._repick_job = self.root.after(80, self.repick)

    def repick(self, smooth_only=False):
        """Re-pick the selected picked piece with the current Reach / Smoothing."""
        self._repick_job = None
        i = self.selected
        if i is None or i >= len(self.pieces) or self.alpha is None:
            return
        piece = self.pieces[i]
        if piece.seed is None or piece.edited:
            return
        reach = piece.reach if smooth_only else self.reach.get()
        smooth = self.smooth.get()
        seed = core.transform_points(self.R, piece.seed[None])[0]
        poly = self._pick(seed, reach, smooth)
        if poly is None:
            return
        piece.poly = core.transform_points(self.Rinv, poly)
        piece.reach, piece.smooth = reach, smooth
        self._update_rows()
        self.render()
        self.status.set(f"Piece {i + 1}: reach {reach}, smoothing {smooth}")

    def set_pick_mode(self, on):
        """Auto select piece mode: a left click picks the object under it; dragging pans."""
        if on and self.cropping:
            on = False
        if on and self.select_mode.get():
            self.set_select_mode(False)
        if on and self.edit_mode:
            self.set_edit_mode(None)
        self.pick_mode.set(on)
        self._pick_press = None
        self.pick_button.state(["pressed"] if on else ["!pressed"])
        self.pick_button.configure(text="Auto selecting (Esc to stop)" if on else "Auto select piece")
        if not self.cropping:
            self.canvas.config(cursor=self._mode_cursor())
        if self.flat is not None and not self.cropping:
            self.status.set("Auto select piece: click a drawing's line or a solid shape. Esc stops."
                            if on else "Drag to move, scroll to zoom.")
        self._draw_overlays()

    def save_piece(self, i, p):
        if not self._ready():
            return False
        return self._write(p, lambda: self._piece_rgba(i))

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
            messagebox.showinfo("No pieces", "Click Lasso piece and drag around a drawing, or Auto select piece and click one.",
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
        if not self.pieces:     # Tk keeps an emptied frame's size otherwise
            self.list_inner.configure(height=1)
        self._rows, self._thumbs = [], []
        if not hasattr(self, "_blank"):  # sizes empty thumbnails in pixels, not characters
            self._blank = tk.PhotoImage(width=THUMB, height=THUMB)
        for i in range(len(self.pieces)):
            bg = palette()["row_sel"] if i == self.selected else palette()["row"]
            row = tk.Frame(self.list_inner, bg=bg, padx=4, pady=4)
            row.pack(fill="x", pady=2)
            col = tk.Frame(row, bg=bg)        # thumbnail with + / − under it
            col.pack(side="left", anchor="n")
            thumb = tk.Label(col, bg=bg, image=self._blank, width=THUMB, height=THUMB)
            thumb.pack()
            edit_box = tk.Frame(col, bg=bg)
            edit_box.pack(pady=(2, 0))
            for op, label, tip in (("add", "+", EDIT_TIPS["add"]), ("cut", "−", EDIT_TIPS["cut"])):
                b = ttk.Button(edit_box, text=label, width=3,
                               command=lambda i=i, op=op: self.toggle_edit_mode(i, op))
                self._tip(b, tip).pack(side="left", padx=1)
                if self.edit_mode == (i, op):
                    b.state(["pressed"])
            info = tk.Frame(row, bg=bg)
            info.pack(side="left", fill="both", expand=True, padx=(6, 0))
            text = tk.Label(info, text=f"Piece {i + 1}", bg=bg, fg=palette()["fg"],
                            anchor="w", justify="left")
            text.pack(anchor="w")
            btns = tk.Frame(info, bg=bg)
            btns.pack(anchor="w", pady=(2, 0))
            for label, cmd, tip in (
                    ("Open…", self.edit_piece, "Open this piece in its own window to adjust it"),
                    ("Save…", self.save_piece_dialog, "Save this piece as a PNG"),
                    ("✕ Remove", self.delete_piece, "Delete this piece")):
                self._tip(ttk.Button(btns, text=label, width=11,
                                     command=lambda i=i, cmd=cmd: cmd(i)),
                          tip).pack(anchor="w", pady=(2, 0))
            for widget in (row, thumb, text, info, btns, col, edit_box):
                widget.bind("<Button-1>", lambda e, i=i: self.select_piece(i))
            self._rows.append((row, thumb, text, info, btns, col, edit_box))
            self._thumbs.append(None)
        # An emptied list doesn't always send <Configure>; fix the scroll region now
        self.list_inner.update_idletasks()
        self._update_list_region()

    def _update_rows(self):
        """Refresh piece thumbnails and sizes from the current result."""
        if self.alpha is None:
            return
        for i in range(min(len(self.pieces), len(self._rows))):
            out = self._piece_rgba(i)
            h, w = out.shape[:2]
            k = min(THUMB / w, THUMB / h)
            small = cv2.resize(out, (max(int(w * k), 1), max(int(h * k), 1)),
                               interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
            img = ImageTk.PhotoImage(Image.fromarray(self._preview_rgb(small)))
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
        self.canvas.config(cursor=self._mode_cursor())

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
        if self.edit_mode:
            self._edit_with_loop(pts)
            return
        if not pts or len(pts) < 3:
            return
        pts = np.array(pts, np.float32)
        if cv2.contourArea(pts) * self.scale ** 2 < 64:  # a click or a scribble
            return
        h, w = self.flat.shape
        pts[:, 0] = pts[:, 0].clip(0, w - 1)
        pts[:, 1] = pts[:, 1].clip(0, h - 1)
        pts = pts.round().astype(np.int32)
        note, v = "", self.smooth.get()
        hugged = core.hug_loop(self._hug_alpha(), pts, smooth=v, color=self.color) if self.hug.get() else None
        if hugged is not None:
            i = self.add_piece(hugged, loop=pts, hugged=True, smooth=v)
            note = " (hugging the drawing)"
        else:
            if self.hug.get():
                note = " (no ink found inside the loop, kept as drawn)"
            i = self.add_piece(pts, loop=pts)
        self.status.set(f"Added piece {i + 1}{note}. Drag around another drawing, or save.")

    def _edit_with_loop(self, pts):
        """+ Add area / − Cut area: change the target piece with the loop just
        drawn (working coords). In add mode, a click inside another piece adds
        that piece's whole area."""
        i, op = self.edit_mode
        if not pts:
            return
        click = len(pts) < 3 or cv2.contourArea(np.array(pts, np.float32)) * self.scale ** 2 < 64
        h, w = self.flat.shape
        if click:
            if op != "add":
                return
            x, y = pts[-1]
            hits = [j for j in range(len(self.pieces))
                    if j != i and core.shape_contains(self.piece_poly(j), (x, y))]
            if not hits:
                self.status.set("Click inside another piece, or drag a loop around an area.")
                return
            j = hits[-1]                     # the one drawn on top
            other, what = self.piece_poly(j), f"piece {j + 1}"
        else:
            loop = np.array(pts, np.float32)
            loop[:, 0] = loop[:, 0].clip(0, w - 1)
            loop[:, 1] = loop[:, 1].clip(0, h - 1)
            other = loop.round().astype(np.int32)
            what = "the loop"
            if op == "add" and self.hug.get():
                hugged = core.hug_loop(self._hug_alpha(), other, smooth=self.smooth.get(),
                                       color=self.color)
                if hugged is not None:
                    other, what = hugged, "the drawing in the loop"
        result = core.combine_shapes(self.piece_poly(i), other, op, (h, w))
        if result is None:
            self.status.set("That would remove the whole piece; use ✕ Remove to delete it.")
            return
        piece = self.pieces[i]
        piece.poly = [core.transform_points(self.Rinv, c.astype(np.float64)) for c in result]
        piece.edited, piece.loop, piece.seed = True, None, None
        self._update_rows()
        self.render()
        verb = "Added" if op == "add" else "Cut"
        prep = "to" if op == "add" else "from"
        self.status.set(f"{verb} {what} {prep} piece {i + 1}. Keep going, or press Esc.")

    def toggle_edit_mode(self, i, op):
        on = self.edit_mode != (i, op)
        self.set_edit_mode((i, op) if on else None)

    def set_edit_mode(self, mode):
        """mode = (piece index, "add" | "cut"), or None to stop. While on,
        left-drag draws loops that change that piece."""
        if mode and (self.cropping or self.flat is None):
            mode = None
        self._lasso_cancel()
        if mode:
            if self.select_mode.get():
                self.set_select_mode(False)
            if self.pick_mode.get():
                self.set_pick_mode(False)
        self.edit_mode = mode
        if mode:
            self.selected = mode[0]
        self._rebuild_rows()
        self._update_rows()
        if not self.cropping:
            self.canvas.config(cursor=self._mode_cursor())
        if mode:
            i, op = mode
            self.status.set(f"Adding to piece {i + 1}: drag a loop, or click another piece. "
                            "Esc stops." if op == "add" else
                            f"Cutting from piece {i + 1}: drag a loop around what to remove. "
                            "Esc stops.")
        elif self.flat is not None and not self.cropping:
            self.status.set("Drag to move, scroll to zoom.")
        self.render()

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
        self.color = self.rcolor[y:y + h, x:x + w]
        self.valid = self.rvalid[y:y + h, x:x + w]
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
        self._close_other_sections("crop")
        self._flush()
        self.set_select_mode(False)
        self.set_pick_mode(False)
        if self.edit_mode:
            self.set_edit_mode(None)
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
        self.crop_button.state(["pressed"])
        self._update_spin_limits()
        self._sync_spins()
        self.fit = True
        self.status.set("Crop: drag the box's edges or type values, then Save crop (Enter) "
                        "or click Crop again. Reset puts the box back at the edges.")
        self.refresh()

    def toggle_crop(self):
        """The Crop button opens the crop section; clicking it again keeps the box."""
        if self.cropping:
            self.save_crop()
        else:
            self.start_crop()

    def reset_crop(self):
        """Put the crop box back at the image edges (and untick "to the drawing")."""
        if not self.cropping:
            return
        self.set_crop_box(*self._full_rect())
        self.crop_fit.set(False)
        self._fit_backup = None
        self.status.set("Crop box reset to the whole image.")

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

    def _leave_crop_mode(self):
        if not self.cropping:
            return
        self.cropping = False
        self._crop_drag = None
        self._fit_backup = None
        self.crop_section.pack_forget()
        self.crop_holder.configure(height=1)   # Tk keeps an emptied frame's size otherwise
        self.crop_button.state(["!pressed"])
        if not self.tips.winfo_manager():   # back under the specks slider
            self.tips.pack(anchor="w", pady=(10, 0), after=self.specks_scale.master)
        self.canvas.config(cursor=self._mode_cursor())
        self.fit = True
        self._draw_overlays()

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
        cursor = self._CROP_CURSORS.get(self._crop_hit(e.x, e.y), "arrow")
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

    # --------------------------------------------------------- mouse modes

    def _mode_cursor(self):
        if self.pick_mode.get():
            return "tcross"
        return "crosshair" if self.select_mode.get() or self.edit_mode else "fleur"

    def set_select_mode(self, on):
        """Lasso piece mode: left-drag draws loops. Off: left-drag pans."""
        if on and self.cropping:     # the left button belongs to the crop box right now
            on = False
        if not on:
            self._lasso_cancel()
        elif self.pick_mode.get():
            self.set_pick_mode(False)
        if on and self.edit_mode:
            self.set_edit_mode(None)
        self.select_mode.set(on)
        self.select_button.state(["pressed"] if on else ["!pressed"])
        self.select_button.configure(text="Lassoing pieces (Esc to stop)" if on
                                     else "Lasso piece")
        if not self.cropping:
            self.canvas.config(cursor=self._mode_cursor())
        if self.flat is not None and not self.cropping:
            self.status.set("Lasso piece: drag around a drawing. Esc stops." if on
                            else "Drag to move, scroll to zoom.")
        self._draw_overlays()

    def _left_press(self, e):
        if self.cropping:
            self._crop_press(e)
        elif self.select_mode.get() or self.edit_mode:
            self._lasso_start(e)
        else:
            if self.pick_mode.get():
                self._pick_press = (e.x, e.y)
            self._pan_start(e)

    def _left_move(self, e):
        if self.cropping:
            self._crop_move(e)
        elif self._lasso:
            self._lasso_move(e)
        elif self._pan:
            self._pan_move(e)

    def _left_release(self, e):
        if self.cropping:
            self._crop_release(e)
        elif self._lasso:
            self._lasso_end(e)
        elif self._pan:
            press, self._pick_press = self._pick_press, None
            self._pan_end(e)
            if press and self.pick_mode.get() and math.hypot(e.x - press[0], e.y - press[1]) < 6:
                self.pick_at(*self.to_image(e.x, e.y))

    def _on_escape(self, _e=None):
        if self.cropping:            # like clicking Crop again: keeps the box
            self.save_crop()
        elif self._lasso:
            self._lasso_cancel()
        elif self.edit_mode:
            self.set_edit_mode(None)
        elif self.select_mode.get():
            self.set_select_mode(False)
        elif self.pick_mode.get():
            self.set_pick_mode(False)

    def _on_return(self, _e=None):
        if self.cropping:
            self.save_crop()

    def _background(self, w, h, mode=None, cell=12):
        mode = mode or self.bg_mode.get()
        key = (mode, w, h, cell)
        if key not in self._bg_cache:
            if len(self._bg_cache) > 50:
                self._bg_cache.clear()
            if mode == "Dark":
                bg = np.full((h, w, 3), 40, np.float32)
            elif mode == "Checker":
                yy, xx = np.indices((h, w))
                cells = ((yy // cell + xx // cell) % 2).astype(np.float32)
                bg = np.repeat((150 + 60 * cells)[:, :, None], 3, axis=2)
            else:
                bg = np.full((h, w, 3), 245, np.float32)
            self._bg_cache[key] = bg
        return self._bg_cache[key]

    def _composite(self, alpha, bg=None):
        """Alpha -> RGB preview: tint color over the preview background."""
        h, w = alpha.shape
        a = (alpha.astype(np.float32) / 255)[:, :, None]
        bg = self._background(w, h) if bg is None else bg
        out = bg * (1 - a) + np.array(self.tint, np.float32) * a
        return out.astype(np.uint8)

    def _composite_color(self, bgr, alpha, bg=None):
        """The photo (with its alpha) over the preview background, as RGB."""
        h, w = alpha.shape
        a = (alpha.astype(np.float32) / 255)[:, :, None]
        bg = self._background(w, h) if bg is None else bg
        out = bg * (1 - a) + bgr[:, :, ::-1].astype(np.float32) * a
        return out.astype(np.uint8)

    def _preview_rgb(self, bgra):
        """A saved BGRA image as a piece thumbnail: always over a fine checkerboard,
        so transparent parts are easy to see."""
        h, w = bgra.shape[:2]
        bg = self._background(w, h, mode="Checker", cell=6)
        if self.extract.get():
            return self._composite(bgra[:, :, 3], bg)
        return self._composite_color(bgra[:, :, :3], bgra[:, :, 3], bg)

    def _mode_banner_text(self):
        """What the banner over the preview says, or None when no mode is on."""
        if self.cropping:
            return "Cropping (Enter to save, Esc to close)"
        if self.edit_mode:
            i, op = self.edit_mode
            if op == "add":
                return f"Adding to piece {i + 1}: loop an area or click another piece (Esc to stop)"
            return f"Cutting from piece {i + 1}: loop what to remove (Esc to stop)"
        if self.pick_mode.get():
            return "Auto selecting pieces (Esc to stop)"
        if self.select_mode.get():
            return "Lassoing pieces (Esc to stop)"
        return None

    def _draw_banner(self):
        """A see-through dark strip across the top of the preview naming the mode."""
        self.canvas.delete("banner")
        text = self._mode_banner_text()
        if text is None or self.flat is None:
            return
        cw = max(self.canvas.winfo_width(), 1)
        strip = Image.new("RGBA", (cw, BANNER_H), (0, 0, 0, 150))
        self._banner_photo = ImageTk.PhotoImage(strip)
        self.canvas.create_image(0, 0, anchor="nw", image=self._banner_photo,
                                 tags="banner", state="disabled")
        self.canvas.create_text(cw / 2, BANNER_H / 2, text=text, fill="white",
                                font=("Segoe UI", 11, "bold"), tags="banner", state="disabled")

    def _draw_overlays(self):
        self._draw_banner()
        self._show_opts()

    # -------------------------------------------------------- piece settings

    def _visible_opts(self):
        """Which piece options matter in the current mode (none = no box)."""
        if self.cropping or self.flat is None:
            return []
        if self.edit_mode:
            return ["hug", "smooth"] if self.edit_mode[1] == "add" else []
        if self.pick_mode.get():
            return ["reach", "smooth"]
        if self.select_mode.get():
            return ["hug", "smooth"]
        return []

    def _show_opts(self):
        """Show only the piece settings that matter right now in their slot
        above Auto select piece (the slot keeps its size, so nothing jumps)."""
        if not hasattr(self, "_opt_frames"):
            return
        keys = self._visible_opts()
        if keys == self._opts_shown:
            return
        self._opts_shown = keys
        for f in self._opt_frames.values():
            f.pack_forget()
        for n, k in enumerate(keys):     # padding inside the highlight; the last gets the bottom
            last = n == len(keys) - 1
            self._opt_frames[k].pack(fill="x", padx=OPT_PADX,
                                     pady=(0, OPT_PADY if last else 0))
        # Highlighted in the "on" color of the button these settings belong to
        self.opts_frame.configure(style="Opts.TFrame" if keys else "TFrame")

    def _view_crop(self, arr):
        """The visible part of a working-image layer, resized to screen pixels.
        Returns (layer, x0, y0) with (x0, y0) in photo coords, or None."""
        cw, ch = self._canvas_size()
        H, W = arr.shape[:2]
        s = self.scale
        x0, y0 = max(math.floor(self.vx), 0), max(math.floor(self.vy), 0)
        x1 = min(math.ceil(self.vx + cw / s), W)
        y1 = min(math.ceil(self.vy + ch / s), H)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = arr[y0:y1, x0:x1]
        tw, th = max(round((x1 - x0) * s), 1), max(round((y1 - y0) * s), 1)
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
        return cv2.resize(crop, (tw, th), interpolation=interp), x0, y0

    def _view_alpha(self):
        # Resize the *result*, not the photo: shrinking the photo first would
        # blend thin lines into the paper until they fell below White.
        return self._view_crop(self.alpha)

    def render(self):
        """Draw the visible part of the result at the current zoom."""
        if self.alpha is None:
            return
        s = self.scale
        self.canvas.delete("all")
        if self.extract.get():
            view = self._view_alpha()
            if view is None:
                return
            shown, x0, y0 = view
            img = self._composite(shown)
        else:
            view = self._view_crop(self.color)
            if view is None:
                return
            shown, x0, y0 = view
            img = self._composite_color(shown, self._view_crop(self.valid)[0])
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
            poly = self.piece_poly(i)
            contours = [(c - (self.vx, self.vy)) * s
                        for c in (poly if isinstance(poly, list) else [poly])]
            sel = i == self.selected
            color = SELECTED_COLOR if sel else PIECE_COLOR
            for pts in contours:
                self.canvas.create_polygon(pts.ravel().tolist(), outline=color, fill="",
                                           width=3 if sel else 2)
            pts = np.concatenate(contours)
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
        self._draw_overlays()


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
