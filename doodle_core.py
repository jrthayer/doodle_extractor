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


def _contours(shape):
    """A piece shape as a list of int32 (N, 1, 2) contours. A shape is either
    one polygon ((N, 2) points) or a list of contours (outlines and holes),
    as made by `combine_shapes`."""
    if isinstance(shape, (list, tuple)) and len(shape) and np.ndim(shape[0]) >= 2:
        return [np.asarray(c, np.int32).reshape(-1, 1, 2) for c in shape]
    return [np.asarray(shape, np.int32).reshape(-1, 1, 2)]


def _all_points(shape):
    return np.concatenate([c.reshape(-1, 2) for c in _contours(shape)])


def combine_shapes(a, b, op, size):
    """Join (`op="add"`) or cut (`op="cut"`) shape `b` into shape `a`. `size`
    is the image's (h, w). Returns a list of contours (outlines and holes), or
    None if nothing is left."""
    H, W = size
    pts = np.concatenate([_all_points(a), _all_points(b)])
    x0, y0 = max(int(pts[:, 0].min()) - 2, 0), max(int(pts[:, 1].min()) - 2, 0)
    x1, y1 = min(int(pts[:, 0].max()) + 3, W), min(int(pts[:, 1].max()) + 3, H)
    if x1 <= x0 or y1 <= y0:
        return None

    def raster(shape):
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [c - (x0, y0) for c in _contours(shape)], 255)
        return m

    ma, mb = raster(a), raster(b)
    m = cv2.bitwise_or(ma, mb) if op == "add" else cv2.bitwise_and(ma, cv2.bitwise_not(mb))
    cs, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c.reshape(-1, 2) + (x0, y0) for c in cs if len(c) >= 3 and cv2.contourArea(c) >= 4]
    return cs or None


def shape_contains(shape, pt):
    """True if `pt` (x, y) is inside the shape (holes count as outside)."""
    p = (float(pt[0]), float(pt[1]))
    n = sum(cv2.pointPolygonTest(c, p, False) >= 0 for c in _contours(shape))
    return n % 2 == 1


def cut_piece(alpha, polygon, pad):
    """Keep only the ink inside `polygon` (list of (x, y) points, or a list of
    contours), cropped to that ink plus `pad`. The edge is anti-aliased."""
    H, W = alpha.shape
    contours = _contours(polygon)
    pts = _all_points(polygon)

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
    cv2.fillPoly(region, [c - (x0, y0) for c in contours], 255, lineType=cv2.LINE_AA)
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
    pts = _all_points(polygon)
    x0 = max(int(pts[:, 0].min()) - margin, 0)
    y0 = max(int(pts[:, 1].min()) - margin, 0)
    x1 = min(int(pts[:, 0].max()) + margin + 1, W)
    y1 = min(int(pts[:, 1].max()) + margin + 1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    region = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(region, [c - (x0, y0) for c in _contours(polygon)], 255,
                 lineType=cv2.LINE_AA)
    return (x0, y0, x1, y1), region


def _loop_object(bgr, inside, band):
    """Pixels inside the loop whose color differs clearly from the background,
    learned from the strip just inside the loop (`band`)."""
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    samples = lab[band].reshape(-1, 3).astype(np.float32)
    if len(samples) < 8:
        return np.zeros(inside.shape, bool)
    k = min(4, len(samples))
    _, labels, centers = cv2.kmeans(samples, k, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                                    3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    share = np.bincount(labels, minlength=k) / len(samples)
    # Background = the most common color along the loop, plus colors close to it
    # (e.g. paper that's lit unevenly). Strokes the loop crosses stay "object".
    bg = [int(share.argmax())]
    grew = True
    while grew:
        grew = False
        for c in range(k):
            if c not in bg and share[c] >= 0.05 and min(
                    np.linalg.norm(centers[c] - centers[b]) for b in bg) < 25:
                bg.append(c)
                grew = True
    centers_bg = centers[bg]

    def nearest(px):
        return np.sqrt(((px[..., None, :] - centers_bg) ** 2).sum(-1)).min(-1)

    bg_samples = samples[np.isin(labels, bg)]
    thr = max(30.0, 2 * float(np.percentile(nearest(bg_samples), 90)))
    return (nearest(lab) > thr) & inside


def _drop_clipped(blob, band):
    """Drop slivers of neighbouring drawings that the loop cuts through. Keeps
    the biggest part, every part clear of the loop, and big parts touching it."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    if n <= 2:
        return blob
    areas = stats[1:, cv2.CC_STAT_AREA]
    touching = np.zeros(n, bool)
    touching[np.unique(labels[band & (labels > 0)])] = True
    keep = np.zeros(n, bool)
    keep[1:] = ~touching[1:] | (areas >= 0.25 * areas.max())
    keep[1 + int(areas.argmax())] = True
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _is_background(lab, region):
    """Is `region` the color of the picture's background (e.g. the hole of an O,
    or the paper between lines)? The background is learned from the border."""
    h, w = region.shape
    b = max(2, round(0.03 * max(h, w)))
    strip = np.ones((h, w), bool)
    strip[b:-b, b:-b] = False
    samples = lab[strip].reshape(-1, 3).astype(np.float32)
    k = min(3, len(samples))
    _, labels, centers = cv2.kmeans(samples, k, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                                    2, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(labels.ravel(), minlength=k) / len(samples)
    mean = lab[region].mean(axis=0)
    return any(np.linalg.norm(c - mean) < 20 for c in centers[share >= 0.15])


def pick_object(alpha, color, seed, reach=50, smooth=0, ink=64):
    """Find the object at `seed` (x, y) and return its outline as an int32
    (N, 2) polygon, or None if there's nothing there.

    A click on a line picks the strokes connected to it; `reach` (0-100) also
    pulls in strokes across gaps (100 bridges about as much as hugging does).
    A click inside a solid shape picks that shape (magic-wand style)."""
    H, W = alpha.shape
    gap = max(3, round(0.02 * max(H, W)))
    s = min(1.0, 10 / gap)
    sw, sh = max(1, round(W * s)), max(1, round(H * s))
    gs = max(1, round(gap * s))
    ink_mask = (alpha >= ink).astype(np.uint8) * 255
    cand = cv2.resize(ink_mask, (sw, sh), interpolation=cv2.INTER_AREA) > 0
    sx = min(max(int(seed[0] * s), 0), sw - 1)
    sy = min(max(int(seed[1] * s), 0), sh - 1)

    flood = None
    if color is not None:
        lab = cv2.cvtColor(cv2.resize(color, (sw, sh), interpolation=cv2.INTER_AREA)
                           .astype(np.float32) / 255, cv2.COLOR_BGR2Lab)

        def wand(x, y):
            """Magic wand: the region of similar color around (x, y), and what it
            is: "shape" (a distinct object), "hole" (enclosed background, like
            the inside of an O) or "background"."""
            fm = np.zeros((sh + 2, sw + 2), np.uint8)
            cv2.floodFill(lab, fm, (x, y), 0, (12, 25, 25), (12, 25, 25),
                          4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
            region = fm[1:-1, 1:-1] > 0
            edge = region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()
            if region.mean() > 0.25 or edge:
                return "background", region
            return ("hole" if _is_background(lab, region) else "shape"), region

        kind, region = wand(sx, sy)
        if kind == "hole":   # clicked inside a hole: pick the shape around it
            rim = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            rim &= ~region
            ys_, xs_ = np.nonzero(rim)
            if len(xs_):
                far = np.linalg.norm(lab[ys_, xs_] - lab[region].mean(axis=0), axis=1)
                k = int(far.argmax())
                sx, sy = int(xs_[k]), int(ys_[k])
                kind, region = wand(sx, sy)
        if kind == "shape":
            cand |= region
            flood = region

    if not cand[sy, sx]:   # clicked next to the object: snap to the nearest part
        r = max(2, round(0.015 * max(H, W) * s))
        ys, xs = np.nonzero(cand[max(sy - r, 0):sy + r + 1, max(sx - r, 0):sx + r + 1])
        if len(xs) == 0:
            return None
        k = int(np.argmin((xs + max(sx - r, 0) - sx) ** 2 + (ys + max(sy - r, 0) - sy) ** 2))
        sx, sy = int(xs[k]) + max(sx - r, 0), int(ys[k]) + max(sy - r, 0)

    # A solid shape (the magic wand got most of what was clicked) is already
    # whole: Reach only bridges to other shapes above 50. Line drawings need
    # their strokes bridged, so Reach applies from 0.
    reach = max(0, min(reach, 100))
    solid = False
    if flood is not None:
        _, lab0 = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
        part = lab0 == lab0[sy, sx]
        solid = (flood & part).sum() >= 0.5 * part.sum()
    g = round((max(0, reach - 50) / 50 if solid else reach / 100) * gs)

    # Group: bridge gaps up to the reach, keep the part the click is in
    p = g + 2 * gs + 8
    grid = cv2.copyMakeBorder(cand.astype(np.uint8) * 255, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    if g > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k)
    _, labels = cv2.connectedComponents(grid, connectivity=8)
    comp = (labels == labels[sy + p, sx + p]).astype(np.uint8) * 255
    if g > 0:   # the real pixels of that part (not the bridges)
        comp &= cv2.copyMakeBorder(cand.astype(np.uint8) * 255, p, p, p, p,
                                   cv2.BORDER_CONSTANT, value=0)

    # Auto loop a little way around it, so the strip inside the loop is background
    m = 2 * gs + 4
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * m + 1, 2 * m + 1))
    around = cv2.dilate(cv2.morphologyEx(comp, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * gs + 1, 2 * gs + 1))) if g == 0 else comp, k)
    contours, _ = cv2.findContours(around, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    loop = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64) - p
    loop = (loop + 0.5) / s - 0.5
    loop[:, 0] = loop[:, 0].clip(0, W - 1)
    loop[:, 1] = loop[:, 1].clip(0, H - 1)
    loop = loop.round().astype(np.int32)

    # Only this part (with its inside) counts as drawing, so reach decides what's
    # in, not the loop's margin
    shape = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * max(g, 1) + 1, 2 * max(g, 1) + 1)))
    outer, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape = np.zeros_like(shape)
    cv2.drawContours(shape, outer, -1, 255, cv2.FILLED)
    only_small = cv2.dilate(shape, np.ones((5, 5), np.uint8))[p:-p, p:-p]
    only = cv2.resize(only_small, (W, H), interpolation=cv2.INTER_NEAREST) > 0
    hugged = hug_loop(alpha, loop, ink=ink, smooth=smooth, color=color, only=only)
    return hugged if hugged is not None else loop


def hug_loop(alpha, polygon, ink=64, smooth=0, color=None, only=None):
    """Shrink a hand-drawn loop to the outline of the drawing inside it.

    Nearby strokes are joined, holes are filled (the inside of the drawing is
    kept) and separate parts end up in one outline. A small margin is added,
    but the result never leaves the original loop. `smooth` (0-100) rounds the
    outline out without cutting into the drawing. With `color` (the BGR image
    in the same coords) solid and colored shapes count too, judged against the
    background just inside the loop. Slivers of neighbours that the loop cuts
    through are left out. Returns an int32 (N, 2) polygon in the same coords,
    or None if there's nothing inside to hug."""
    bounds = _loop_bounds(alpha.shape, polygon, 0)
    if bounds is None:
        return None
    (x0, y0, x1, y1), region = bounds
    inside = region >= 128
    blob = ((alpha[y0:y1, x0:x1] >= ink) & inside).astype(np.uint8) * 255

    H, W = alpha.shape
    gap = max(3, round(0.02 * max(H, W)))     # bridge gaps this wide (px)
    # Join and fill at a reduced scale (gap ~10 px there): wide gaps stay fast.
    s = min(1.0, 10 / gap)
    full_size = (blob.shape[1], blob.shape[0])
    sub_color = None if color is None else color[y0:y1, x0:x1]
    small_inside = inside
    if s < 1:
        small = (max(1, round(full_size[0] * s)), max(1, round(full_size[1] * s)))
        blob = (cv2.resize(blob, small, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255
        small_inside = cv2.resize(inside.astype(np.uint8), small,
                                  interpolation=cv2.INTER_NEAREST) > 0
        if sub_color is not None:
            sub_color = cv2.resize(sub_color, small, interpolation=cv2.INTER_AREA)
        gap = max(1, round(gap * s))

    # The strip just inside the loop: the loop goes AROUND the drawing, so it's background
    band = small_inside & ~(cv2.erode(small_inside.astype(np.uint8), np.ones((7, 7), np.uint8),
                                      borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0)
    if sub_color is not None:   # solid / colored shapes the ink detection only sees faintly
        blob[_loop_object(sub_color, small_inside, band)] = 255
    if only is not None:        # caller says which area may count as the drawing
        sub_only = only[y0:y1, x0:x1].astype(np.uint8)
        if s < 1:
            sub_only = cv2.resize(sub_only, (blob.shape[1], blob.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        blob[sub_only == 0] = 0
    if cv2.countNonZero(blob) < max(3, 20 * s * s):
        return None
    blob = _drop_clipped(blob, band)

    def fill_only(img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(img)
        cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
        return out, contours

    def close_and_fill(img, g):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(img)
        cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
        return out, contours

    # Smoothing radius at the working scale (0-100 -> up to 4x the bridged gap)
    r = round(max(0, min(smooth, 100)) / 100 * 4 * gap)
    sigma = max(1.0, r / 2)
    # Pad so closing / blurring can reach across the box edges too
    p = max(4 * gap + 2, r + math.ceil(3 * sigma) + 2)
    work = cv2.copyMakeBorder(blob, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    # Specks would force wide bridging below: drop them first
    n, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    if n > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.r_[False, areas >= 0.005 * areas.max()]
        work = np.where(keep[labels], 255, 0).astype(np.uint8)

    # Solid shapes (a filled letter, a painted object) are already one piece:
    # trace them as they are, so notches like a W's valleys survive.
    solid = False
    for g in (0, 1, 2, 4, 6):
        filled, contours = close_and_fill(work, g) if g else fill_only(work)
        if len(contours) == 1:
            solid = cv2.countNonZero(work & filled) >= 0.6 * cv2.countNonZero(filled)
            break
    if not solid:
        # Line drawings: bridge the gaps between strokes so they become one shape
        filled, contours = close_and_fill(work, gap)
        g = gap
        for _ in range(3):             # still in parts: join wider gaps
            if len(contours) <= 1:
                break
            g *= 2
            filled, contours = close_and_fill(filled, min(g, p // 2))
    if len(contours) > 1:              # far apart: one outline around all of them
        hull = cv2.convexHull(np.vstack(contours))
        filled = np.zeros_like(filled)
        cv2.fillPoly(filled, [hull], 255)
    if r > 0:
        # Round it out. Only ever grows (OR with the original), so no ink is lost:
        # closing fills the bays, the blur softens the jitter. Solid shapes only
        # get a lighter blur, so their real notches stay.
        if solid:
            soft = cv2.GaussianBlur(filled, (0, 0), sigma / 2)
        else:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            soft = cv2.GaussianBlur(cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k), (0, 0), sigma)
        filled = np.where((soft >= 128) | (filled > 0), 255, 0).astype(np.uint8)
    filled = filled[p:-p, p:-p]
    if s < 1:
        filled = cv2.resize(filled, full_size, interpolation=cv2.INTER_LINEAR)
        filled = (filled >= 128).astype(np.uint8) * 255

    m = max(2, round(1 / s) + 2)       # a little room around the lines (covers the rescale)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * m + 1, 2 * m + 1))
    filled = cv2.dilate(filled, k)
    filled[~inside] = 0                # never outside the user's loop

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    outline = contours[0] if len(contours) == 1 else cv2.convexHull(np.vstack(contours))
    outline = cv2.approxPolyDP(outline, 1.5, True).reshape(-1, 2)
    if len(outline) < 3:
        return None
    return (outline + (x0, y0)).astype(np.int32)


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
