import sys
import cv2

import numpy as np

#load image from command line
img = cv2.imread(sys.argv[1])

#img fails to load
if img is None:
    print("Image failed to load!")
    exit()

print("image loaded!")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray.png", img)

print("image grayscaled!")

# --- Find the sheet of paper ---
blur = cv2.GaussianBlur(gray, (0, 0), 5)        # smooth out the grid/ink/noise
_, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sheet = max(contours, key=cv2.contourArea)       # biggest bright shape = the page

mask = np.zeros_like(gray)
cv2.drawContours(mask, [sheet], -1, 255, thickness=cv2.FILLED)
mask = cv2.erode(mask, np.ones((15, 15), np.uint8))   # pull in past the shadowy edge
cv2.imwrite("step0_mask.png", mask)              # check it: white = page

# --- Replace everything off the page with the paper's color (your GIMP fill) ---
paper_value = int(np.median(gray[mask == 255]))
gray[mask == 0] = paper_value

#remove paper
bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
flat = cv2.divide(gray, bg, scale=255)
cv2.imwrite("step2_flat.png", flat)

#seperate ink from everything else
# - Grid ghosts still visible? Lower WHITE.
# - Strokes look thin or broken? WHITE is too low. Raise it, carefully.
# - Stroke middles look gray instead of solid white? Raise BLACK.
# - A few faint grid fragments left? Step 5's speck removal will clean those up.

BLACK = 80 #ink color
WHITE = 160 #everything else

f = flat.astype(np.float32)
alpha = (WHITE - f) / (WHITE - BLACK)   # 1.0 at BLACK, 0.0 at WHITE
alpha = np.clip(alpha, 0, 1)
alpha = (alpha * 255).astype(np.uint8)

cv2.imwrite("step3_alpha.png", alpha)

alpha[mask == 0] = 0

#build png
h, w = alpha.shape
rgba = np.full((h, w, 4), 255, dtype=np.uint8)   # all white, fully solid
rgba[:, :, 3] = alpha                             # replace the 4th channel with your alpha
cv2.imwrite("step4_out.png", rgba)