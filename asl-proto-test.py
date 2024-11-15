import numpy as np
import cv2
import sys
import mediapipe as mp
from enum import Enum
import math


# font parameters
fontFamily    = cv2.FONT_HERSHEY_SIMPLEX
fontScale     = 1
fontThickness = 2
def displayText(frame, text, position):
    frame = cv2.putText(frame, text, position, fontFamily, fontScale, (0,0,0), fontThickness*2, cv2.LINE_AA)
    frame = cv2.putText(frame, text, position, fontFamily, fontScale, (255,255,255), fontThickness, cv2.LINE_AA)

# helper math functions

def slope(point1, point2):
    return (point2.y - point1.y) / (point2.x - point1.x)

def angle(point1, point2):
    return math.atan2(point2.y - point1.y, point2.x - point1.x)

def approx(value, target, range):
    return (target - range <= value) and (value <= target + range) 



class Finger(Enum):
    STRAIGHT = 1
    ANGLE    = 2
    UP       = 3
    DOWN     = 4
    LEFT     = 5
    RIGHT    = 6
    BENT     = 7
    PALM     = 8
    STRAIGHT_UP    = 9
    STRAIGHT_DOWN  = 10
    STRAIGHT_LEFT  = 11
    STRAIGHT_RIGHT = 12
    
# finger 0 is thumb, finger 1 is index, etc.
def analyzeFinger(fingerIndex, fingerConfig, landmark):
    # i: joint index.
    i = 1 + 4 * fingerIndex
    if fingerConfig == Finger.STRAIGHT:
        angle1 = angle(landmark[i+0], landmark[i+1]) + 600.28
        angle2 = angle(landmark[i+1], landmark[i+2]) + 600.28
        angle3 = angle(landmark[i+2], landmark[i+3]) + 600.28
        range = math.radians(30)

        return (approx(angle1, angle2, range) or approx(angle1 + 2*math.pi, angle2, range) or approx(angle1, angle2 + 2*math.pi, range)) \
           and (approx(angle2, angle3, range) or approx(angle2 + 2*math.pi, angle3, range) or approx(angle2, angle3 + 2*math.pi, range))
    # finger tip angle
    elif fingerConfig == Finger.ANGLE:
        return angle(landmark[i+2], landmark[i+3])
    elif fingerConfig == Finger.UP:
        a = angle(landmark[i+2], landmark[i+3])
        return -3 * 3.14 / 4 < a and a < -1 * 3.14 / 4
    elif fingerConfig == Finger.DOWN:
        a = angle(landmark[i+2], landmark[i+3])
        return 1 * 3.14 / 4 < a and a < 3 * 3.14 / 4
    elif fingerConfig == Finger.LEFT:
        a = angle(landmark[i+2], landmark[i+3])
        return a < -3 * 3.1415926 / 4 or a > 3 * 3.1415926 / 4
    elif fingerConfig == Finger.RIGHT:
        a = angle(landmark[i+2], landmark[i+3])
        return -1 * 3.14 / 4 < a and a < 1 * 3.14 / 4
    # assumes hand direction up
    elif fingerConfig == Finger.BENT:
        return (landmark[i+3].y > landmark[i+2].y) and (landmark[i+3].y < landmark[i+0].y)
    # assumes hand direction up
    elif fingerConfig == Finger.PALM:
        return landmark[i+3].y > landmark[i+0].y
    elif fingerConfig == Finger.STRAIGHT_UP:
        return analyzeFinger(fingerIndex, Finger.STRAIGHT, landmark) \
           and analyzeFinger(fingerIndex, Finger.UP, landmark)
    elif fingerConfig == Finger.STRAIGHT_LEFT:
        return analyzeFinger(fingerIndex, Finger.STRAIGHT, landmark) \
           and analyzeFinger(fingerIndex, Finger.LEFT, landmark)
    else:
        return None


def determineLetter(landmark, hand_type):
    if analyzeFinger(0, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(1, Finger.PALM, landmark) \
        and analyzeFinger(2, Finger.PALM, landmark) \
        and analyzeFinger(3, Finger.PALM, landmark) \
        and analyzeFinger(4, Finger.PALM, landmark):
        return "A"
    elif analyzeFinger(1, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(2, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(3, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(4, Finger.STRAIGHT_UP, landmark) \
        and ((hand_type == "Right" and landmark[4].x > landmark[8].x) or (hand_type == "Left" and landmark[4].x < landmark[8].x)): # thumb tip crosses index finger
        return "B"
    elif (hand_type == "Right" \
        and analyzeFinger(1, Finger.LEFT, landmark) \
        and analyzeFinger(2, Finger.LEFT, landmark) \
        and analyzeFinger(3, Finger.LEFT, landmark) \
        and analyzeFinger(4, Finger.LEFT, landmark) \
        and analyzeFinger(0, Finger.LEFT, landmark)) or \
        (hand_type == "Left" \
        and analyzeFinger(1, Finger.RIGHT, landmark) \
        and analyzeFinger(2, Finger.RIGHT, landmark) \
        and analyzeFinger(3, Finger.RIGHT, landmark) \
        and analyzeFinger(4, Finger.RIGHT, landmark) \
        and analyzeFinger(0, Finger.RIGHT, landmark)): # TODO: distinguish from "O"
        return "C"
    elif analyzeFinger(1, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(2, Finger.PALM, landmark) \
        and analyzeFinger(3, Finger.PALM, landmark) \
        and analyzeFinger(4, Finger.PALM, landmark) \
        and landmark[4].x > landmark[8].x: # thumb tip crosses index finger # LEFT D confused with I
        return "D"
    elif (analyzeFinger(1, Finger.BENT, landmark) or analyzeFinger(1, Finger.PALM, landmark)) \
        and (analyzeFinger(2, Finger.BENT, landmark) or analyzeFinger(2, Finger.PALM, landmark)) \
        and (analyzeFinger(3, Finger.BENT, landmark) or analyzeFinger(3, Finger.PALM, landmark)) \
        and (analyzeFinger(4, Finger.BENT, landmark) or analyzeFinger(4, Finger.PALM, landmark)) \
        and ((hand_type == "Right" and analyzeFinger(0, Finger.RIGHT, landmark)) or (hand_type == "Left" and analyzeFinger(0, Finger.LEFT, landmark))):
        return "E" 
    elif analyzeFinger(2, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(3, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(4, Finger.STRAIGHT_UP, landmark) \
        and not analyzeFinger(1, Finger.STRAIGHT, landmark): # no condition on thumb, too restrictive
        return "F" # TODO: add parallel condition on fingers 2/3/4? closeness of 0/1?
    elif analyzeFinger(1, Finger.STRAIGHT, landmark) \
        and (hand_type == "Right" and analyzeFinger(1, Finger.LEFT, landmark) or hand_type == "Left" and analyzeFinger(1, Finger.RIGHT, landmark)) \
        and not analyzeFinger(2, Finger.STRAIGHT, landmark) \
        and not analyzeFinger(3, Finger.STRAIGHT, landmark) \
        and not analyzeFinger(4, Finger.STRAIGHT, landmark) \
        and landmark[4].y > landmark[8].y: # avoid "gun" gesture recognized as "G"
        return "G"
    elif analyzeFinger(1, Finger.STRAIGHT, landmark) \
        and (hand_type == "Right" and analyzeFinger(1, Finger.LEFT, landmark) or hand_type == "Left" and analyzeFinger(1, Finger.RIGHT, landmark)) \
        and analyzeFinger(2, Finger.STRAIGHT, landmark) \
        and (hand_type == "Right" and analyzeFinger(2, Finger.LEFT, landmark) or hand_type == "Left" and analyzeFinger(2, Finger.RIGHT, landmark)) \
        and not analyzeFinger(3, Finger.STRAIGHT, landmark) \
        and not analyzeFinger(4, Finger.STRAIGHT, landmark) \
        and landmark[4].y > landmark[8].y: # avoid "gun" gesture recognized as "H"
        return "H"
    elif analyzeFinger(1, Finger.PALM, landmark) \
        and analyzeFinger(2, Finger.PALM, landmark) \
        and analyzeFinger(3, Finger.PALM, landmark) \
        and analyzeFinger(4, Finger.STRAIGHT, landmark) \
        and (landmark[8].x < landmark[4].x and landmark[4].x < landmark[20].x) or (landmark[20].x < landmark[4].x and landmark[4].x < landmark[8].x):
        return "I" # thump tip between index and pinky tip # THIS IS BROKEN
    elif analyzeFinger(0, Finger.STRAIGHT, landmark) \
        and analyzeFinger(1, Finger.PALM, landmark) \
        and analyzeFinger(2, Finger.PALM, landmark) \
        and analyzeFinger(3, Finger.PALM, landmark) \
        and analyzeFinger(4, Finger.STRAIGHT, landmark):
        return "Y"
    else:
        return "?"


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
    summary = "no data"

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0] # just first hand
        mp_drawing.draw_landmarks(frame, hand_landmarks,  mp_hands.HAND_CONNECTIONS)
        landmark = result.multi_hand_landmarks[0].landmark
        hand_type = result.multi_handedness[0].classification[0].label # "Left" or "Right"
        summary = determineLetter(result.multi_hand_landmarks[0].landmark, hand_type)
        # summary = analyzeFinger(1, Finger.ANGLE, landmark)
        summary = str(summary)
        
        """"
        # to process multiple hands
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks,  mp_hands.HAND_CONNECTIONS)
            # landmark: 0 for palm, 4 for thump tip, 8 for index tip
            # see https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker for numbering scheme
            landmark = result.multi_hand_landmarks[0].landmark
            summary = determineLetter(result.multi_hand_landmarks[0].landmark)
            # summary = analyzeFinger(1, Finger.ANGLE, landmark)
            summary = str(summary)
        """
    # draw text on image
    displayText(frame, summary, (50,50) )



    cv2.imshow("press Q to quit", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

