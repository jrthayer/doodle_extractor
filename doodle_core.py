"""Doodle -> Godot ink sprite pipeline.

Turns a photo of a hand drawing into an RGBA PNG whose color is pure white
and whose alpha is the ink darkness, so it can be tinted in Godot with
`self_modulate`.

The work is split into a slow part (`prepare`, done once per photo) and a
fast part (`make_alpha` and friends, rerun whenever a setting changes).
"""

import math
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


# ---------------------------------------------------------------- rotation
# Angles are in degrees CLOCKWISE (as seen on screen). `quarter` counts
# 90-degree clockwise turns, which are done losslessly.

_ROTATE_CODES = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180,
                 3: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _fine_matrix(w, h, angle):
    """Affine matrix rotating a w x h image by `angle` clockwise, with the
    canvas grown so nothing is cut off. Returns (2x3 matrix, (new_w, new_h))."""
    cx, cy = (w - 1) / 2, (h - 1) / 2
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)  # OpenCV: positive = counter-clockwise
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(math.ceil(h * sin + w * cos - 1e-9))
    nh = int(math.ceil(h * cos + w * sin - 1e-9))
    M[0, 2] += (nw - 1) / 2 - cx
    M[1, 2] += (nh - 1) / 2 - cy
    return M, (nw, nh)


def rotation_matrix(shape, quarter=0, angle=0.0):
    """3x3 matrix mapping original photo (x, y) to rotated-photo (x, y),
    matching `rotate_prepared`. Returns (matrix, (new_w, new_h))."""
    h, w = shape
    q = quarter % 4
    if q == 0:
        A, size = [[1, 0, 0], [0, 1, 0]], (w, h)
    elif q == 1:   # clockwise: (x, y) -> (h-1-y, x)
        A, size = [[0, -1, h - 1], [1, 0, 0]], (h, w)
    elif q == 2:
        A, size = [[-1, 0, w - 1], [0, -1, h - 1]], (w, h)
    else:          # counter-clockwise: (x, y) -> (y, w-1-x)
        A, size = [[0, 1, 0], [-1, 0, w - 1]], (h, w)
    A = np.vstack([np.array(A, np.float64), [0, 0, 1]])
    if angle:
        F, size = _fine_matrix(size[0], size[1], angle)
        A = np.vstack([F, [0, 0, 1]]) @ A
    return A, size


def rotate_image(img, quarter=0, angle=0.0, border=0, interp=cv2.INTER_LINEAR):
    """Rotate a 1- or 3-channel image (lossless for whole quarter turns, then
    `angle` degrees clockwise). New corners are filled with `border`."""
    if img is None:
        return None
    q = quarter % 4
    if q:
        img = cv2.rotate(img, _ROTATE_CODES[q])
    if angle:
        h, w = img.shape[:2]
        M, size = _fine_matrix(w, h, angle)
        border = (border,) * 3 if img.ndim == 3 else border
        img = cv2.warpAffine(img, M, size, flags=interp, borderValue=border)
    return img


def rotate_prepared(flat, mask, quarter=0, angle=0.0):
    """Rotate a prepared photo. New corners become paper (transparent)."""
    return (rotate_image(flat, quarter, angle, 255, cv2.INTER_LINEAR),
            rotate_image(mask, quarter, angle, 0, cv2.INTER_NEAREST))


def rotate_color(bgr, valid, quarter=0, angle=0.0):
    """Rotate a color photo and its alpha (`valid`, None = fully opaque).
    New corners become transparent. Returns (bgr, valid)."""
    if valid is None:
        valid = np.full(bgr.shape[:2], 255, np.uint8)
    return (rotate_image(bgr, quarter, angle, 0, cv2.INTER_LINEAR),
            rotate_image(valid, quarter, angle, 0, cv2.INTER_LINEAR))


def transform_points(matrix, pts):
    """Apply a 3x3 (or 2x3) affine matrix to an (N, 2) array of points."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return pts @ matrix[:2, :2].T + matrix[:2, 2]


def rect_corners(rect):
    """Centres of the 4 corner pixels of an (x, y, w, h) rectangle."""
    x, y, w, h = rect
    return np.array([(x, y), (x + w - 1, y), (x + w - 1, y + h - 1), (x, y + h - 1)],
                    np.float64)


def rect_from_corners(matrix, corners, size):
    """Bounding rectangle (x, y, w, h) of `corners` after `matrix`, clamped
    to an image of `size` = (width, height). Inverse of `rect_corners`
    when the matrix is the identity or a 90-degree turn."""
    W, H = size
    pts = transform_points(matrix, corners)
    x0 = min(max(int(round(pts[:, 0].min())), 0), W - 1)
    y0 = min(max(int(round(pts[:, 1].min())), 0), H - 1)
    x1 = min(max(int(round(pts[:, 0].max())), x0), W - 1)
    y1 = min(max(int(round(pts[:, 1].max())), y0), H - 1)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


# ------------------------------------------------ fast: per setting change

def levels_lut(black, white):
    """256-entry table: <= black -> 255, >= white -> 0, soft between."""
    black, white = int(black), int(white)
    if white <= black:
        white = black + 1
    f = np.arange(256, dtype=np.float32)
    alpha = np.clip((white - f) / (white - black), 0, 1)
    return (alpha * 255).astype(np.uint8)


def make_alpha(flat, mask, black, white):
    """Map darkness to alpha. A lookup table gives the same result as doing
    the math per pixel, but fast enough for full-size live previews."""
    alpha = cv2.LUT(flat, levels_lut(black, white))
    if mask is not None:
        alpha = cv2.bitwise_and(alpha, mask)  # mask is 0 or 255
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
    rows = np.flatnonzero(alpha.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(alpha.any(axis=0))
    x, y = int(cols[0]), int(rows[0])
    w, h = int(cols[-1]) - x + 1, int(rows[-1]) - y + 1
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


def cut_piece(alpha, polygon, pad):
    """Keep only the ink inside `polygon` (list of (x, y) points), cropped
    to that ink plus `pad`. The polygon edge is anti-aliased."""
    H, W = alpha.shape
    pts = np.asarray(polygon, np.int32).reshape(-1, 2)

    # Only work inside the loop's bounding box (plus padding room): much
    # faster than masking the whole photo.
    x0 = max(int(pts[:, 0].min()) - pad - 1, 0)
    y0 = max(int(pts[:, 1].min()) - pad - 1, 0)
    x1 = min(int(pts[:, 0].max()) + pad + 2, W)
    y1 = min(int(pts[:, 1].max()) + pad + 2, H)
    if x1 <= x0 or y1 <= y0:  # loop is entirely off the image
        return np.zeros((1, 1), np.uint8)

    local = (pts - (x0, y0)).reshape(-1, 1, 2)
    sub = alpha[y0:y1, x0:x1]
    region = np.zeros_like(sub)
    cv2.fillPoly(region, [local], 255, lineType=cv2.LINE_AA)
    cut = np.minimum(sub, region)

    box = crop_box(cut, pad)
    if box is None:  # no ink inside: fall back to the loop's own bounds
        sh, sw = cut.shape
        x, y, w, h = cv2.boundingRect(local)
        x, y = min(max(x, 0), sw - 1), min(max(y, 0), sh - 1)
        box = (x, y, max(min(w, sw - x), 1), max(min(h, sh - y), 1))
    x, y, w, h = box
    return cut[y:y + h, x:x + w]


def process_piece(flat, mask, s: Settings, polygon):
    """Output image for one lassoed piece."""
    return to_rgba(cut_piece(finish_alpha(flat, mask, s), polygon, s.pad))


def piece_source(flat, mask, polygon, margin=24):
    """A prepared image containing only the inside of `polygon`, for editing
    a piece on its own. Outside the loop becomes plain paper (255), with a
    soft edge; `margin` px of paper is kept around the loop's bounds.
    Returns (flat, mask) like `prepare`."""
    bounds = _loop_bounds(flat.shape[:2], polygon, margin)
    if bounds is None:
        return np.full((1, 1), 255, np.uint8), None
    (x0, y0, x1, y1), region = bounds
    ink = 255 - flat[y0:y1, x0:x1].astype(np.uint16)          # darkness
    sub = (255 - ink * region // 255).astype(np.uint8)       # fade to paper outside
    sub_mask = None if mask is None else mask[y0:y1, x0:x1].copy()
    return sub, sub_mask


def _loop_bounds(shape, polygon, margin):
    """The loop's bounding box (plus `margin`, clamped to the image) and the
    loop filled in (soft-edged) over that box. None if it's off the image."""
    H, W = shape
    pts = np.asarray(polygon, np.int32).reshape(-1, 2)
    x0 = max(int(pts[:, 0].min()) - margin, 0)
    y0 = max(int(pts[:, 1].min()) - margin, 0)
    x1 = min(int(pts[:, 0].max()) + margin + 1, W)
    y1 = min(int(pts[:, 1].max()) + margin + 1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    region = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(region, [(pts - (x0, y0)).reshape(-1, 1, 2)], 255, lineType=cv2.LINE_AA)
    return (x0, y0, x1, y1), region


# ------------------------------------------------- unfiltered (color) output

def color_rgba(bgr, valid):
    """A color image as BGRA for saving, with `valid` as its alpha."""
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    out[:, :, 3] = 255 if valid is None else valid
    return out


def color_source(bgr, valid, polygon, margin=24):
    """The color counterpart of `piece_source`: the same box, with alpha =
    inside the loop (soft edge) and inside the image. Returns (bgr, valid)."""
    bounds = _loop_bounds(bgr.shape[:2], polygon, margin)
    if bounds is None:
        return np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.uint8)
    (x0, y0, x1, y1), region = bounds
    sub_valid = region if valid is None else np.minimum(region, valid[y0:y1, x0:x1])
    return bgr[y0:y1, x0:x1].copy(), sub_valid


def color_piece(bgr, valid, polygon):
    """A lassoed piece of the color photo as BGRA: transparent outside the
    loop, cropped to the loop's bounds."""
    sub, sub_valid = color_source(bgr, valid, polygon, margin=0)
    return color_rgba(sub, sub_valid)


def process(flat, mask, s: Settings):
    """Full output image from a prepared photo."""
    alpha = finish_alpha(flat, mask, s)
    if s.auto_crop:
        box = crop_box(alpha, s.pad)
        if box is not None:
            x, y, w, h = box
            alpha = alpha[y:y + h, x:x + w]
    return to_rgba(alpha)
