import numpy as np
import cv2
import sys
import mediapipe as mp

import math

print("running webcam app")

# -1 causes error: Camera index out of range
# DSHOW = direct show, used on windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

print("video capture initialized")

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

if not cap.isOpened():
    print("Error opening video")

while True:
    # note: frames loaded in BGR order
    success, frame = cap.read()
    
    if not success:
        cap.release()
        print("Error reading video")
        sys.exit()

    # reflect across axis 0 (x-axis)
    frame = cv2.flip(frame, 1)
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(RGB_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # print(hand_landmarks)
            mp_drawing.draw_landmarks(frame, hand_landmarks,  mp_hands.HAND_CONNECTIONS)
            # landmark: 0 for palm, 4 for thump tip, 8 for index tip
            # see https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker for numbering scheme
            hand0 = result.multi_hand_landmarks[0]
            thumbTip = hand0.landmark[4]
            indexTip = hand0.landmark[8]
            print(thumbTip)



    cv2.imshow("press Q to quit", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

