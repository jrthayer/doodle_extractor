"""Doodle -> Godot ink sprite pipeline.

Turns a photo of a hand drawing into an RGBA PNG whose color is pure white
and whose alpha is the ink darkness, so it can be tinted in Godot with
`self_modulate`.

The work is split into a slow part (`prepare`, done once per photo) and a
fast part (`make_alpha` and friends, rerun whenever a setting changes).
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class Settings:
    black: int = 80          # this dark or darker -> fully solid ink
    white: int = 160         # this light or lighter -> fully transparent
    min_speck: int = 0       # drop ink blobs smaller than this many px (0 = off)
    auto_crop: bool = False  # trim the output to the ink plus `pad`
    pad: int = 8


# ---------------------------------------------------------------- loading

def load_image(path):
    """Load any image as BGR uint8.

    Uses Pillow so that non-ASCII Windows paths work and phone photos are
    rotated according to their EXIF data. Transparent areas are put onto
    black, so paper detection treats them as "not paper".
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            im = im.convert("RGBA")
            backdrop = Image.new("RGBA", im.size, (0, 0, 0, 255))
            im = Image.alpha_composite(backdrop, im)
        rgb = np.array(im.convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])  # RGB -> BGR for OpenCV


def save_png(path, rgba):
    """Write an RGBA image to a PNG (works with non-ASCII paths)."""
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise IOError("PNG encoding failed")
    buf.tofile(str(path))


# ------------------------------------------------------- slow: per photo

def find_paper(gray):
    """Return a mask of the sheet of paper (255 = page), or None if no
    clear sheet was found (then the whole image is treated as paper)."""
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    sheet = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [sheet], -1, 255, thickness=cv2.FILLED)
    mask = cv2.erode(mask, np.ones((15, 15), np.uint8))  # skip the shadowy edge

    if cv2.countNonZero(mask) < 0.2 * mask.size:
        return None
    return mask


def prepare(bgr, detect_paper=True):
    """Grayscale, paper detection and lighting flattening.

    Returns (flat, mask): `flat` has paper close to 255 and ink dark;
    `mask` is 255 on the page, or None if the whole image counts.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    mask = find_paper(gray) if detect_paper else None
    if mask is not None:
        # Fill everything off the page with the paper's own color
        paper_value = int(np.median(gray[mask == 255]))
        gray[mask == 0] = paper_value

    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    flat = cv2.divide(gray, bg, scale=255)
    return flat, mask


# ------------------------------------------------ fast: per setting change

def make_alpha(flat, mask, black, white):
    """Map darkness to alpha: <= black -> 255, >= white -> 0, soft between."""
    black, white = int(black), int(white)
    if white <= black:
        white = black + 1
    f = flat.astype(np.float32)
    alpha = np.clip((white - f) / (white - black), 0, 1)
    alpha = (alpha * 255).astype(np.uint8)
    if mask is not None:
        alpha[mask == 0] = 0
    return alpha


def remove_specks(alpha, min_area):
    """Erase connected ink blobs smaller than `min_area` pixels."""
    if min_area <= 0:
        return alpha
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (alpha > 0).astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # label 0 is the background
    return np.where(keep[labels], alpha, 0).astype(np.uint8)


def crop_box(alpha, pad):
    """Bounding box (x, y, w, h) of the ink plus padding, or None if empty."""
    pts = cv2.findNonZero(alpha)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    H, W = alpha.shape
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    return x0, y0, x1 - x0, y1 - y0


def to_rgba(alpha):
    """White everywhere, with `alpha` as the transparency."""
    h, w = alpha.shape
    rgba = np.full((h, w, 4), 255, dtype=np.uint8)
    rgba[:, :, 3] = alpha
    return rgba


def finish_alpha(flat, mask, s: Settings):
    """Alpha after levels and speck removal (no cropping)."""
    alpha = make_alpha(flat, mask, s.black, s.white)
    return remove_specks(alpha, s.min_speck)


def process(flat, mask, s: Settings):
    """Full output image from a prepared photo."""
    alpha = finish_alpha(flat, mask, s)
    if s.auto_crop:
        box = crop_box(alpha, s.pad)
        if box is not None:
            x, y, w, h = box
            alpha = alpha[y:y + h, x:x + w]
    return to_rgba(alpha)
