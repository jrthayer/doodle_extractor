"""Doodle Converter: windowed app.

Open a photo of a drawing, tune the sliders while watching the preview,
then save a Godot-ready PNG (white lines, transparent everywhere else).
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import doodle_core as core

IMAGE_TYPES = [("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
               ("All files", "*.*")]
BACKGROUNDS = ("Light", "Dark", "Checker")


class DoodleApp:
    def __init__(self, root):
        self.root = root
        root.title("Doodle Converter")
        root.geometry("1100x750")
        root.minsize(700, 450)

        self.path = None          # current photo
        self.bgr = None           # full-res photo
        self.flat = None          # full-res prepared image
        self.mask = None
        self.pflat = None         # preview-size copies
        self.pmask = None
        self.pscale = 1.0         # preview size / full size
        self.tint = (0, 0, 0)     # preview line color (RGB)
        self._photo = None        # keep Tk image alive
        self._redraw_job = None
        self._resize_job = None
        self._bg_cache = {}

        self.black = tk.IntVar(value=80)
        self.white = tk.IntVar(value=160)
        self.min_speck = tk.IntVar(value=0)
        self.pad = tk.IntVar(value=8)
        self.auto_crop = tk.BooleanVar(value=False)
        self.detect_paper = tk.BooleanVar(value=True)
        self.bg_mode = tk.StringVar(value="Light")

        self._build_ui()
        root.bind("<Control-o>", lambda e: self.open_dialog())
        root.bind("<Control-s>", lambda e: self.save_dialog())

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
        bg.bind("<<ComboboxSelected>>", lambda e: self.schedule_redraw())

        ttk.Label(bar, text="Preview line color:").pack(side="left")
        self.tint_swatch = tk.Button(bar, width=3, relief="solid", bd=1,
                                     bg=self._hex(self.tint), command=self.pick_tint)
        self.tint_swatch.pack(side="left", padx=4)

        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True)

        side = ttk.Frame(body, padding=(10, 6))
        side.pack(side="left", fill="y")
        self._slider(side, "Black (solid ink at or below)", self.black, 0, 255, self.on_black)
        self._slider(side, "White (transparent at or above)", self.white, 0, 255, self.on_white)
        self._slider(side, "Remove specks smaller than (px)", self.min_speck, 0, 2000,
                     lambda v: self.schedule_redraw())

        ttk.Separator(side).pack(fill="x", pady=10)
        ttk.Checkbutton(side, text="Detect paper (ignore the desk)",
                        variable=self.detect_paper, command=self.reprepare).pack(anchor="w")
        ttk.Checkbutton(side, text="Crop to the drawing", variable=self.auto_crop,
                        command=self.schedule_redraw).pack(anchor="w", pady=(6, 0))
        self._slider(side, "Crop padding (px)", self.pad, 0, 200,
                     lambda v: self.schedule_redraw())

        ttk.Separator(side).pack(fill="x", pady=10)
        ttk.Button(side, text="Reset settings", command=self.reset).pack(anchor="w")
        tip = ("Tips:\n"
               "• Grid showing? Lower White.\n"
               "• Lines thin or broken? Raise White.\n"
               "• Line middles gray? Raise Black.\n\n"
               "The saved PNG is always white;\n"
               "set the color in Godot with\n"
               "Self Modulate.")
        ttk.Label(side, text=tip, foreground="#555", justify="left").pack(anchor="w", pady=(12, 0))

        self.canvas = tk.Canvas(body, bg="#808080", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_resize)
        self._show_hint()

        self.status = tk.StringVar(value="Open a photo to start (Ctrl+O).")
        ttk.Label(self.root, textvariable=self.status, anchor="w",
                  padding=(8, 3), relief="sunken").pack(side="bottom", fill="x")

    def _slider(self, parent, label, var, lo, hi, command):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(8, 0))
        tk.Scale(parent, from_=lo, to=hi, orient="horizontal", variable=var,
                 length=240, command=command).pack(anchor="w")

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
                             min_speck=self.min_speck.get(),
                             auto_crop=self.auto_crop.get(), pad=self.pad.get())

    def on_black(self, _=None):
        if self.black.get() >= self.white.get():
            self.white.set(min(self.black.get() + 1, 255))
            self.black.set(self.white.get() - 1)
        self.schedule_redraw()

    def on_white(self, _=None):
        if self.white.get() <= self.black.get():
            self.black.set(max(self.white.get() - 1, 0))
            self.white.set(self.black.get() + 1)
        self.schedule_redraw()

    def reset(self):
        d = core.Settings()
        self.black.set(d.black)
        self.white.set(d.white)
        self.min_speck.set(d.min_speck)
        self.pad.set(d.pad)
        self.auto_crop.set(d.auto_crop)
        if not self.detect_paper.get():
            self.detect_paper.set(True)
            self.reprepare()
        self.schedule_redraw()

    def pick_tint(self):
        rgb, _ = colorchooser.askcolor(color=self._hex(self.tint), title="Preview line color")
        if rgb:
            self.tint = tuple(int(c) for c in rgb)
            self.tint_swatch.configure(bg=self._hex(self.tint))
            self.schedule_redraw()

    # ---------------------------------------------------------- open / save

    def open_dialog(self):
        initial = str(self.path.parent) if self.path else None
        p = filedialog.askopenfilename(title="Open drawing", filetypes=IMAGE_TYPES,
                                       initialdir=initial)
        if p:
            self.open_path(p)

    def open_path(self, p):
        p = Path(p)
        try:
            self._busy(f"Loading {p.name}…")
            self.bgr = core.load_image(p)
        except Exception as e:
            self._idle()
            messagebox.showerror("Couldn't open image", f"{p}\n\n{e}")
            return False
        self.path = p
        self.root.title(f"Doodle Converter: {p.name}")
        self.reprepare()
        return True

    def reprepare(self):
        """Run the slow part of the pipeline for the current photo."""
        if self.bgr is None:
            return
        self._busy("Processing…")
        try:
            self.flat, self.mask = core.prepare(self.bgr, self.detect_paper.get())
        finally:
            self._idle()
        h, w = self.flat.shape
        paper = "page detected" if self.mask is not None else "no separate page found"
        self.status.set(f"{self.path.name}: {w}×{h}, {paper}")
        self.make_preview()

    def save_dialog(self):
        if self.flat is None:
            messagebox.showinfo("Nothing to save", "Open a photo first.")
            return
        p = filedialog.asksaveasfilename(
            title="Save ink PNG", defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialdir=str(self.path.parent), initialfile=self.path.stem + "_ink.png")
        if p:
            self.save_path(p)

    def save_path(self, p):
        p = Path(p)
        try:
            self._busy("Saving…")
            rgba = core.process(self.flat, self.mask, self.settings())
            core.save_png(p, rgba)
        except Exception as e:
            self._idle()
            messagebox.showerror("Couldn't save", f"{p}\n\n{e}")
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

    # -------------------------------------------------------------- preview

    def on_resize(self, _event):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(100, self.make_preview)

    def make_preview(self):
        """Shrink the prepared image to fit the canvas."""
        self._resize_job = None
        if self.flat is None:
            self._show_hint()
            return
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        h, w = self.flat.shape
        scale = min((cw - 20) / w, (ch - 20) / h, 1.0)
        pw, ph = max(int(w * scale), 1), max(int(h * scale), 1)
        self.pscale = scale
        if scale < 1.0:
            self.pflat = cv2.resize(self.flat, (pw, ph), interpolation=cv2.INTER_AREA)
            self.pmask = (None if self.mask is None else
                          cv2.resize(self.mask, (pw, ph), interpolation=cv2.INTER_NEAREST))
        else:
            self.pflat, self.pmask = self.flat, self.mask
        self.redraw()

    def schedule_redraw(self, *_):
        if self._redraw_job:
            self.root.after_cancel(self._redraw_job)
        self._redraw_job = self.root.after(40, self.redraw)

    def _background(self, w, h):
        key = (self.bg_mode.get(), w, h)
        if key not in self._bg_cache:
            mode = self.bg_mode.get()
            if mode == "Dark":
                bg = np.full((h, w, 3), 40, np.float32)
            elif mode == "Checker":
                yy, xx = np.indices((h, w))
                cells = ((yy // 12 + xx // 12) % 2).astype(np.float32)
                bg = np.repeat((150 + 60 * cells)[:, :, None], 3, axis=2)
            else:
                bg = np.full((h, w, 3), 245, np.float32)
            self._bg_cache = {key: bg}  # keep only the current one
        return self._bg_cache[key]

    def redraw(self):
        self._redraw_job = None
        if self.pflat is None:
            return
        s = self.settings()
        alpha = core.make_alpha(self.pflat, self.pmask, s.black, s.white)
        if s.min_speck > 0:
            area = max(1, round(s.min_speck * self.pscale ** 2))
            alpha = core.remove_specks(alpha, area)

        h, w = alpha.shape
        a = (alpha.astype(np.float32) / 255)[:, :, None]
        tint = np.array(self.tint, np.float32)
        out = self._background(w, h) * (1 - a) + tint * a
        self._photo = ImageTk.PhotoImage(Image.fromarray(out.astype(np.uint8)))

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        x0, y0 = (cw - w) // 2, (ch - h) // 2
        self.canvas.delete("all")
        self.canvas.create_image(x0, y0, anchor="nw", image=self._photo)

        if s.auto_crop:
            box = core.crop_box(alpha, round(s.pad * self.pscale))
            if box:
                x, y, bw, bh = box
                self.canvas.create_rectangle(x0 + x, y0 + y, x0 + x + bw, y0 + y + bh,
                                             outline="#ff3b6b", dash=(5, 3), width=2)


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
