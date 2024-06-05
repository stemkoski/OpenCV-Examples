import cv2
import sys

# image modes: -1 default, 0 grayscale, 1 transparency?
img = cv2.imread("images/color-grid.png", -1)

# set to pixel dimensions
# img2 = cv2.resize( img, (512, 512) )
# set to scaling value
# img2 = cv2.resize( img, (0,0), fx=0.5, fy=0.25)
# rotate
img2 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow("ImageWindow", img2)

# wait (0 = forever) for next key press; store value of pressed key
k = cv2.waitKey(0)

# check if a particular key was pressed
if k == ord("s"):
    cv2.imwrite("images/color-grid-new.png", img2)

cv2.destroyAllWindows()
