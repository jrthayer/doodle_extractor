import sys
import cv2

#load image from command line
img = cv2.imread(sys.argv[1])

#img fails to load
if img is None:
    print("Image failed to load!")
    exit()

print("image loaded!")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray.png", img)

print("image grayscaled!")

#remove paper
bg = cv2.GaussianBlur(img, (0, 0), sigmaX=25)
flat = cv2.divide(img, bg, scale=255)
cv2.imwrite("step2_flat.png", flat)

#seperate ink from everything else
import numpy as np
BLACK = 80 #ink color
WHITE = 170 #everything else

f = flat.astype(np.float32)
alpha = (WHITE - f) / (WHITE - BLACK)   # 1.0 at BLACK, 0.0 at WHITE
alpha = np.clip(alpha, 0, 1)
alpha = (alpha * 255).astype(np.uint8)

cv2.imwrite("step3_alpha.png", alpha)

#build png
h, w = alpha.shape
rgba = np.full((h, w, 4), 255, dtype=np.uint8)   # all white, fully solid
rgba[:, :, 3] = alpha                             # replace the 4th channel with your alpha
cv2.imwrite("step4_out.png", rgba)