import numpy as np
import cv2
import sys
import time

print("running webcam app")

# -1 causes error: Camera index out of range
# DSHOW = direct show, used on windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("video capture initialized")

if not cap.isOpened():
    print("Error opening video")

while True:
    success, frame = cap.read()
    
    if not success:
        cap.release()
        print("Error reading video")
        sys.exit()

    cv2.imshow("press Q to quit", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

