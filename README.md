# Doodle Extractor

Crops, rotates and cuts drawings out of photos. **Detect lines** turns them into clean lines:
**black on white** by default, or any line color and background you pick. **Apply Godot** gives
Godot-ready PNGs: **white lines, transparent everywhere else**. In Godot, put one on a `Sprite2D`
and set **Self Modulate** to recolor the lines.

## Using the app
Run `dist\DoodleExtractor.exe` (or `python doodle_gui.py`). You can also drag a photo onto the exe.

1. **Open…** (Ctrl+O): pick the photo.
2. By default the photo is shown and saved **as is** (rotated and cropped if you did). Every
   on/off switch in the app starts off.
   Click **Detect lines** to open its settings, then switch on **Apply detection**. The other
   settings appear once it's on, and the button shows **Detect lines ✓** while detection is applied:
   - **Black**: ink this dark or darker is fully solid.
   - **White**: anything this light or lighter is transparent. Lower it if the grid shows.
   - **Remove specks**: erases dots smaller than N pixels.
   - **Detect paper**: ignores the background around the page. Switch it on for photos of a page
     on a desk or table.
   - **Line color** (in the toolbar, shown while detection is applied) is the color of the saved
     lines; black by default.
   - **Apply Godot** (next to Line color) switches to white lines on a transparent background, for
     Godot's Self Modulate. While it's on, Background and Line color are locked; click it again to
     get your own colors back. Switching detection off turns it off too.

   **Background** (in the toolbar) is saved too: **White** (the default), **Black**, **Transparent**
   (shown as a checkerboard) or **Custom…** (pick any color; the swatch next to it changes it). A
   solid background makes the saved PNG opaque. It applies to the photo as well, filling corners
   exposed by straightening and the area outside a doodle's loop.

   Switch **Apply detection** off to go back to the photo; your slider values are kept for next
   time. Clicking **Detect lines** again only hides the settings, so detection stays on (✓).
3. **Save PNG…** (Ctrl+S): choose where to save the image (the cropped area, if you cropped).

**Crop** opens the crop settings and shows a crop box over the photo, starting at the edges.
The button stays pressed while the section is open:
- Drag its edges or corners to resize it, or drag inside to move it. You can also type or step
  the **X / Y / Width / Height** values.
- **Crop to paper** snaps the box tight to the ink on the paper (the background around the page
  is ignored even with Detect paper off) and keeps following it as you change the sliders.
  Switching it off puts the box back where it was.
- **Save crop** (Enter) applies it. Clicking **Crop** again (or Esc) also closes the section
  and keeps your changes.
- **Reset** puts the box back at the image edges (no crop).
- Opening **Crop** again later shows the box where you left it, so you can widen it back out.
- The crop follows the photo when you rotate it. Doodle editor windows start cropped to the drawing.

**Rotate** opens the rotation settings: ⟲ 90° / ⟳ 90° for sideways photos, and **Straighten**
(−45° to +45°) for a slight tilt. Corners exposed by straightening are transparent. Doodles stay
attached to their drawings. Click **Rotate** again to hide the settings; the rotation stays.

**Detect lines**, **Crop** and **Rotate** sit side by side at the top left. Each opens its
settings under the row, highlighted in the same gray as the pressed button. Only one is open at a
time: opening another closes it (closing Crop this way keeps your crop, like clicking Crop again).

**Dark theme:** click the moon at the right of the toolbar (it becomes a sun; click that to go back). The app starts in whichever mode
Windows apps use (Settings > Personalization > Colors), and every open window switches together.

**Moving around:** drag the preview to move the image, scroll to zoom around the cursor.
The toolbar has −, +, **Fit** and **100%**; Ctrl+0 fits, Ctrl+plus and Ctrl+minus zoom.

**Not sure what a button does?** Hover over it: the status bar at the bottom says. Sliders
explain themselves the same way, and the − / + beside each slider step it one notch (hold to
repeat).

### Cutting out doodles
Several drawings on one page? Click **Lasso doodle** (above Save all), then **drag around a drawing**
on the preview. When you let go the loop
closes and the drawing becomes a **doodle** in the panel on the right, while the full photo stays so
you can loop around the next one.
- **Doodle settings** appear in the doodles panel, just above **Auto select doodle**, while you're
  making doodles: **Hug the drawing** and **Smoothing** while lassoing or adding with **+**,
  **Reach** and **Smoothing** while auto selecting.
- **Auto select doodle** (above Lasso doodle): click a drawing instead of looping it. Click one of its
  lines, inside a solid shape, or even inside a hole like the middle of an O. Clicking just next to
  it works too. Dragging still moves the image. **Reach** sets how far it groups nearby strokes
  (pencil drawings are made of separate lines); for solid shapes it only starts grouping above 50.
  Moving Reach or Smoothing re-picks the selected doodle. Drawings whose lines touch are picked
  together; loop around one of them instead.
- **Hug the drawing** (off until you switch it on) pulls the loop in to the drawing's
  outline when you let go. Gaps between strokes are bridged, so separate parts inside one loop stay
  together. It never goes outside your loop. Switch it off to keep loops exactly as you draw them.
  It finds pencil and pen lines as well as solid or colored shapes (anything that stands out from
  the background along your loop), and leaves out slivers of neighbouring drawings your loop clips.
- **Smoothing** rounds the hugged outline out; higher is rounder. It only
  ever grows the outline, so no ink is cut off. Moving it also re-smooths the selected doodle, so you
  can loop first and tune after. Selecting a doodle shows the smoothing it was made with.
- Without line detection a doodle is the photo inside its loop (transparent outside it). With it, the
  doodle is the extracted ink, cropped tight to the drawing.
- **Save…** on a row saves that doodle; **Save all…** writes every doodle to a folder as
  `<photo>_doodle1.png`, `<photo>_doodle2.png`, …
- **Open…** opens the doodle in a new window of its own: full-size preview, zoom, rotation and
  its own settings (starting from the main window's, including whether line detection and Apply Godot are on). **Save PNG** there saves the
  doodle. You can open several at once.
- **+** / **−** (under a doodle's thumbnail) add to or cut from that doodle's outline. With
  **+** on, drag a loop to join it to the doodle (hugged to the drawing when Hug the drawing
  is on), or click inside another doodle to add all of that doodle's area (the other doodle stays
  in the list). With **−** on, drag a loop around what to remove; it cuts exactly where
  you drew, and can leave holes. Both stay on so you can make several changes; Esc or clicking the
  button again stops. Once a doodle has been changed this way, Smoothing and Reach leave it alone.
- Lasso doodle stays on so you can loop several drawings in a row. Click it again (or press
  **Esc**) to go back to moving the image; Esc while drawing cancels just that loop.
- While a mode is on (Auto select doodle, Lasso doodle, + / −, Crop), a dark banner across the top of the
  preview says which one and how to stop it.
- Thumbnails show each doodle as it will be saved, on the chosen background (a checkerboard where
  it's transparent).
- Click a row to select it (highlighted orange on the preview). **Delete** removes the selected doodle.

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
