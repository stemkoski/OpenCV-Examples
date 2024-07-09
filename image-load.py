import cv2
import numpy as np
import sys

print("running...")

# image modes: -1 default, 0 grayscale, 1 transparency?
img = cv2.imread("images/color-grid.png", -1)

# set to pixel dimensions
img2 = cv2.resize( img, (512, 512) )

# set to scaling value
# img2 = cv2.resize( img, (0,0), fx=0.5, fy=0.25)

# rotate
# img2 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# reflect
# 0 = around horizontal axis (swap top and bottom)
# 1 = around vertical axis (swap left and right)
# img2 = cv2.flip(img, 1) 

# convert from BGR to HSV for simpler color masking
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
# get the yellow color mask
mask = cv2.inRange(hsv, (25, 50, 50), (35, 255, 255))
# foreground
fg = cv2.bitwise_and(img2,img2, mask=mask)
# background
mask_inv = cv2.bitwise_not(mask)
bg = cv2.bitwise_and(img2,img2, mask=mask_inv)


cv2.imshow("ImageWindow1", img2)
cv2.imshow("ImageWindow2", mask)
cv2.imshow("ImageWindow3", fg)
cv2.imshow("ImageWindow4", bg)

# Create a blank image
image = np.zeros((512, 512, 3), np.uint8)
# fill image with color
image[:] = (0, 0, 255)
cv2.imshow("ImageWindow5", image)


# wait (0 = forever) for next key press; store value of pressed key
k = cv2.waitKey(0)

# check if a particular key was pressed
if k == ord("s"):
    cv2.imwrite("images/color-grid-new.png", img2)

cv2.destroyAllWindows()
