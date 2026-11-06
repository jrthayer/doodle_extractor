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
   - **Detect paper**: ignores the desk around the page.
   - The preview background and line color are for viewing only. The saved file is always white.
3. **Save PNG…** (Ctrl+S): choose where to save the image (the cropped area, if you cropped).

**Crop…** shows a crop box over the whole photo, starting at the edges:
- Drag its edges or corners to resize it, or drag inside to move it. You can also type or step
  the **X / Y / Width / Height** values.
- **Crop to the drawing** snaps the box tight to the ink and keeps following it as you
  change the sliders. Unticking it puts the box back where it was.
- **Save crop** (Enter) applies it; **Cancel** (Esc) throws the changes away.
- Clicking **Crop…** again later shows the box where you left it, so you can widen it back out.
- The crop follows the photo when you rotate it. Piece editor windows start cropped to the drawing.

**Rotate photo:** ⟲ 90° / ⟳ 90° for sideways photos, and **Straighten** (−45° to +45°) for a slight
tilt. Corners exposed by straightening are transparent. Pieces stay attached to their drawings.

**Zoom:** scroll over the preview to zoom around the cursor, right-drag (or middle-drag) to pan.
The toolbar has −, +, **Fit** and **100%**; Ctrl+0 fits, Ctrl+plus and Ctrl+minus zoom.

### Cutting out pieces
Several drawings on one page? **Drag around a drawing** on the preview. When you let go the loop
closes and the drawing becomes a **piece** in the panel on the right, while the full photo stays so
you can loop around the next one.
- Each piece is cropped tight to the ink inside its loop and follows the
  current slider settings.
- **Save…** on a row saves that piece; **Save all…** writes every piece to a folder as
  `<photo>_piece1.png`, `<photo>_piece2.png`, …
- **Edit…** opens the piece in its own converter window: full-size preview, zoom, rotation and
  its own Black/White settings (starting from the main window's). **Save PNG** there saves the
  piece. You can open several at once.
- Click a row to select it (highlighted orange on the preview). **Delete** removes the selected
  piece, and **Esc** cancels a loop you're drawing.

## Command line
```
python main.py photo.jpg [-o out.png] [--black 80] [--white 160] [--min-speck 50] [--crop] [--pad 8] [--no-paper] [--rotate DEG] [--debug]
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
