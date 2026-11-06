# Doodle Extractor

Crops, rotates and cuts drawings out of photos. **Extract ink** turns them into Godot-ready PNGs:
**white lines, transparent everywhere else**. In Godot, put one on a `Sprite2D` and set
**Self Modulate** to recolor the lines.

## Using the app
Run `dist\DoodleExtractor.exe` (or `python doodle_gui.py`). You can also drag a photo onto the exe.

1. **Open…** (Ctrl+O): pick the photo.
2. By default the photo is shown and saved **as is** (rotated and cropped if you did).
   Click **Extract ink** to turn on ink extraction. The button stays pressed and its settings appear:
   - **Black**: ink this dark or darker is fully solid.
   - **White**: anything this light or lighter is transparent. Lower it if the grid shows.
   - **Remove specks**: erases dots smaller than N pixels.
   - **Detect paper**: ignores the desk around the page.
   - The preview background and line color are for viewing only. The saved ink is always white.

   Click **Extract ink** again to go back to the photo. Your slider values are kept for next time.
3. **Save PNG…** (Ctrl+S): choose where to save the image (the cropped area, if you cropped).

**Crop** opens the crop settings and shows a crop box over the photo, starting at the edges.
The button stays pressed while the section is open:
- Drag its edges or corners to resize it, or drag inside to move it. You can also type or step
  the **X / Y / Width / Height** values.
- **Crop to the drawing** snaps the box tight to the ink and keeps following it as you
  change the sliders. Unticking it puts the box back where it was.
- **Save crop** (Enter) applies it. Clicking **Crop** again (or Esc) also closes the section
  and keeps your changes.
- **Reset** puts the box back at the image edges (no crop).
- Opening **Crop** again later shows the box where you left it, so you can widen it back out.
- The crop follows the photo when you rotate it. Piece editor windows start cropped to the drawing.

**Rotate photo:** ⟲ 90° / ⟳ 90° for sideways photos, and **Straighten** (−45° to +45°) for a slight
tilt. Corners exposed by straightening are transparent. Pieces stay attached to their drawings.

**Moving around:** drag the preview to move the image, scroll to zoom around the cursor.
The toolbar has −, +, **Fit** and **100%**; Ctrl+0 fits, Ctrl+plus and Ctrl+minus zoom.

### Cutting out pieces
Several drawings on one page? Click **Select piece** (above Save all), then **drag around a drawing**
on the preview. When you let go the loop
closes and the drawing becomes a **piece** in the panel on the right, while the full photo stays so
you can loop around the next one.
- Without Extract ink a piece is the photo inside its loop (transparent outside it). With it, the
  piece is the extracted ink, cropped tight to the drawing.
- **Save…** on a row saves that piece; **Save all…** writes every piece to a folder as
  `<photo>_piece1.png`, `<photo>_piece2.png`, …
- **Edit…** opens the piece in its own window: full-size preview, zoom, rotation and
  its own settings (starting from the main window's, including whether Extract ink is on). **Save PNG** there saves the
  piece. You can open several at once.
- Select piece stays on so you can loop several drawings in a row. Click it again (or press
  **Esc**) to go back to moving the image; Esc while drawing cancels just that loop.
- Click a row to select it (highlighted orange on the preview). **Delete** removes the selected piece.

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
pyinstaller --noconfirm --onefile --windowed --name DoodleExtractor doodle_gui.py
```
The result is `dist\DoodleExtractor.exe` (about 70 MB). The first launch takes a few seconds.
Windows Defender sometimes flags homemade PyInstaller exes; that's a known false alarm.
