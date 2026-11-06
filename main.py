"""Command-line version: python main.py photo.jpg [-o out.png] [options]

For the windowed app, run doodle_gui.py instead.
"""

import argparse
from pathlib import Path

import cv2

import doodle_core as core


def main():
    p = argparse.ArgumentParser(description="Convert a doodle photo to a Godot ink sprite.")
    p.add_argument("input", help="photo or scan of the drawing")
    p.add_argument("-o", "--output", help="output PNG (default: <input>_ink.png)")
    p.add_argument("--black", type=int, default=80, help="ink this dark or darker is solid")
    p.add_argument("--white", type=int, default=160, help="this light or lighter is transparent")
    p.add_argument("--min-speck", type=int, default=0, help="remove blobs smaller than N px")
    p.add_argument("--crop", action="store_true", help="crop to the ink")
    p.add_argument("--pad", type=int, default=8, help="padding around the crop")
    p.add_argument("--no-paper", action="store_true", help="skip paper detection")
    p.add_argument("--rotate", type=float, default=0.0,
                   help="rotate the photo by this many degrees clockwise (e.g. 90, -3.5)")
    p.add_argument("--debug", action="store_true", help="also write step images")
    a = p.parse_args()

    src = Path(a.input)
    out = Path(a.output) if a.output else src.with_name(src.stem + "_ink.png")

    try:
        bgr = core.load_image(src)
    except Exception as e:
        raise SystemExit(f"Image failed to load: {e}")
    print("image loaded!")

    flat, mask = core.prepare(bgr, detect_paper=not a.no_paper)
    if a.rotate:
        quarter = round(a.rotate / 90)          # whole turns are lossless
        flat, mask = core.rotate_prepared(flat, mask, quarter, a.rotate - 90 * quarter)
    s = core.Settings(black=a.black, white=a.white, min_speck=a.min_speck,
                      auto_crop=a.crop, pad=a.pad)

    if a.debug:
        if mask is not None:
            cv2.imwrite("step0_mask.png", mask)
        cv2.imwrite("step2_flat.png", flat)
        cv2.imwrite("step3_alpha.png", core.finish_alpha(flat, mask, s))

    core.save_png(out, core.process(flat, mask, s))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
