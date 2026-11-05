# Doodle Converter

Turns a photo of a hand drawing into a Godot-ready PNG: **white lines, transparent everywhere else**.
In Godot, put it on a `Sprite2D` and set **Self Modulate** to recolor the lines.

## Using the app
Run `dist\DoodleConverter.exe` (or `python doodle_gui.py`). You can also drag a photo onto the exe.

1. **Open…** (Ctrl+O): pick the photo.
2. Adjust the sliders while watching the preview:
   - **Black**: ink this dark or darker is fully solid.
   - **White**: anything this light or lighter is transparent. Lower it if the grid shows.
   - **Remove specks**: erases dots smaller than N pixels.
   - **Crop to the drawing** + **padding**: trims the empty page (shown as a dashed box).
   - **Detect paper**: ignores the desk around the page.
   - The preview background and line color are for viewing only. The saved file is always white.
3. **Save PNG…** (Ctrl+S): choose where to save it.

## Command line
```
python main.py photo.jpg [-o out.png] [--black 80] [--white 160] [--min-speck 50] [--crop] [--pad 8] [--no-paper] [--debug]
```
`--debug` also writes the step images (`step0_mask.png`, `step2_flat.png`, `step3_alpha.png`).

## Files
- `doodle_core.py`: the image pipeline (used by both the app and the CLI)
- `doodle_gui.py`: the Windows app (Tkinter)
- `main.py`: command-line version
- `test_core.py`: tests, run with `python test_core.py`

## Packages
- opencv-python: computer vision and image processing library
- numpy: fast numeric arrays. OpenCV stores images as a grid of pixel values using numpy arrays.
- pillow: loads photos (handles phone rotation and non-English file paths) and shows the preview
- pyinstaller: bundles everything into a single `.exe`

## Building the exe
```
pip install -r requirements.txt
pyinstaller --noconfirm --onefile --windowed --name DoodleConverter doodle_gui.py
```
The result is `dist\DoodleConverter.exe` (about 70 MB). The first launch takes a few seconds.
Windows Defender sometimes flags homemade PyInstaller exes; that's a known false alarm.
