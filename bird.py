import cv2
import numpy as np
import sys

print("running...")

# image modes: -1 default, 0 grayscale, 1 transparency?
img = cv2.imread("images/bird.png", -1)

# set to pixel dimensions
img2 = cv2.resize( img, (512, 512) )

# convert from BGR to HSV for simpler color masking
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
# get the white color mask. value of 1 indicates pixel color in range of given values
# for this application, these are the pixels that will be removed
mask_orig = cv2.inRange(hsv, (0, 0, 225), (180, 255, 255))
mask_inv = cv2.bitwise_not(mask_orig)

# "and" with mask: preserve pixels of original image only wherever mask = 1.
postMaskImage = cv2.bitwise_and(img2,img2, mask=mask_inv)

# create a blank image
solidColorImage = np.zeros((512, 512, 3), np.uint8)
# fill blank image with color
solidColorImage[:] = (0, 0, 255)

# apply red to the pixels that were eliminated
solidColorImageSubset = cv2.bitwise_and(solidColorImage,solidColorImage, mask=mask_orig)

newimage = cv2.bitwise_or(postMaskImage, solidColorImageSubset)

cv2.imshow("ImageWindow1", img2)
cv2.imshow("ImageWindow2", mask_orig)
cv2.imshow("ImageWindow3", postMaskImage)
cv2.imshow("ImageWindow4", newimage)



# wait (0 = forever) for next key press; store value of pressed key
k = cv2.waitKey(0)

# check if a particular key was pressed
if k == ord("s"):
    cv2.imwrite("images/color-grid-new.png", img2)

cv2.destroyAllWindows()
