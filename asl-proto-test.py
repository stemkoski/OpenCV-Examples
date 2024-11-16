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

def distance(point1, point2):
    return math.sqrt((point2.y - point1.y) * (point2.y - point1.y) + (point2.x - point1.x) * (point2.x - point1.x))

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
    UP_OR_RIGHT   = 13
    UP_OR_LEFT    = 14
    DOWN_OR_RIGHT = 15
    DOWN_OR_LEFT  = 16
    HAND_FRONT    = 17
    BENT_OR_PALM  = 18
    STRAIGHT_LEFT_OR_RIGHT = 19


# finger 0 is thumb, finger 1 is index, etc.
def analyzeFinger(fingerIndex, fingerConfig, landmark):
    # i: joint index.
    i = 1 + 4 * fingerIndex
    if fingerConfig == Finger.STRAIGHT:
        angle1 = angle(landmark[i+0], landmark[i+1]) + 6.28
        angle2 = angle(landmark[i+1], landmark[i+2]) + 6.28
        angle3 = angle(landmark[i+2], landmark[i+3]) + 6.28
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
    elif fingerConfig == Finger.BENT_OR_PALM: # THIS CAN BE KIND OF MEANINGLESS FROM THE SIDE VIEW - STRAIGHT CAN REGISTER AS PALM IF POINTING SLIGHTLY DOWN
        return analyzeFinger(fingerIndex, Finger.BENT, landmark) \
           or analyzeFinger(fingerIndex, Finger.PALM, landmark)
    elif fingerConfig == Finger.STRAIGHT_UP:
        return analyzeFinger(fingerIndex, Finger.STRAIGHT, landmark) \
           and analyzeFinger(fingerIndex, Finger.UP, landmark)
    elif fingerConfig == Finger.STRAIGHT_LEFT:
        return analyzeFinger(fingerIndex, Finger.STRAIGHT, landmark) \
           and analyzeFinger(fingerIndex, Finger.LEFT, landmark)
    elif fingerConfig == Finger.STRAIGHT_DOWN:
        return analyzeFinger(fingerIndex, Finger.STRAIGHT, landmark) \
           and analyzeFinger(fingerIndex, Finger.DOWN, landmark)
    elif fingerConfig == Finger.STRAIGHT_RIGHT:
        return analyzeFinger(fingerIndex, Finger.STRAIGHT, landmark) \
           and analyzeFinger(fingerIndex, Finger.RIGHT, landmark)
    elif fingerConfig == Finger.STRAIGHT_LEFT_OR_RIGHT:
        return analyzeFinger(fingerIndex, Finger.STRAIGHT, landmark) \
           and (analyzeFinger(fingerIndex, Finger.LEFT, landmark) or analyzeFinger(fingerIndex, Finger.RIGHT, landmark))
    elif fingerConfig == Finger.UP_OR_LEFT:
        return analyzeFinger(fingerIndex, Finger.UP, landmark) \
            or analyzeFinger(fingerIndex, Finger.LEFT, landmark)
    elif fingerConfig == Finger.UP_OR_RIGHT:
        return analyzeFinger(fingerIndex, Finger.UP, landmark) \
            or analyzeFinger(fingerIndex, Finger.RIGHT, landmark)
    elif fingerConfig == Finger.DOWN_OR_LEFT:
        return analyzeFinger(fingerIndex, Finger.DOWN, landmark) \
            or analyzeFinger(fingerIndex, Finger.LEFT, landmark)
    elif fingerConfig == Finger.DOWN_OR_RIGHT:
        return analyzeFinger(fingerIndex, Finger.DOWN, landmark) \
            or analyzeFinger(fingerIndex, Finger.RIGHT, landmark)
    else:
        return None


def determineLetter(landmark, hand_type):
    right_hand = (hand_type == "Right")
    left_hand  = (hand_type == "Left")
    
    if analyzeFinger(0, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(1, Finger.PALM, landmark) \
        and analyzeFinger(2, Finger.PALM, landmark) \
        and analyzeFinger(3, Finger.PALM, landmark) \
        and analyzeFinger(4, Finger.PALM, landmark) \
        and (right_hand and landmark[4].x < landmark[8].x or left_hand and landmark[4].x > landmark[8].x) \
        and (right_hand and landmark[12].x > landmark[5].x or left_hand and landmark[12].x < landmark[5].x):
        return "A" # thumb to correct side of index AND palm facing forward (TODO: flat w.r.t. screen?)
    elif analyzeFinger(0, Finger.UP, landmark) \
        and landmark[4].y < landmark[16].y \
        and (analyzeFinger(1, Finger.BENT_OR_PALM, landmark)) \
        and (analyzeFinger(2, Finger.BENT_OR_PALM, landmark)) \
        and (analyzeFinger(3, Finger.BENT_OR_PALM, landmark)) \
        and (analyzeFinger(4, Finger.BENT_OR_PALM, landmark)) \
        and (right_hand and landmark[6].x  < landmark[4].x and landmark[4].x < landmark[10].x or \
              left_hand and landmark[10].x < landmark[4].x and landmark[4].x < landmark[6].x):
        return "T" # avoid E confusion AND thumb between correct digits
    elif (right_hand and analyzeFinger(0, Finger.RIGHT, landmark) and landmark[4].x > landmark[6].x or \
           left_hand and analyzeFinger(0, Finger.LEFT, landmark)  and landmark[4].x < landmark[6].x) \
        and landmark[4].y < landmark[16].y \
        and (analyzeFinger(1, Finger.BENT_OR_PALM, landmark)) \
        and (analyzeFinger(2, Finger.BENT_OR_PALM, landmark)) \
        and (analyzeFinger(3, Finger.BENT_OR_PALM, landmark)) \
        and (analyzeFinger(4, Finger.BENT_OR_PALM, landmark)):
        return "S" # thumb faces correct horizontal direction AND tip is past correct digits AND avoid confusion with E
    elif landmark[4].y < landmark[16].y \
        and (analyzeFinger(1, Finger.BENT_OR_PALM, landmark)) \
        and (analyzeFinger(2, Finger.BENT_OR_PALM, landmark)) \
        and (analyzeFinger(3, Finger.BENT_OR_PALM, landmark)) \
        and (analyzeFinger(4, Finger.BENT_OR_PALM, landmark)) \
        and (right_hand and landmark[10].x < landmark[4].x and landmark[4].x < landmark[14].x or \
              left_hand and landmark[14].x < landmark[4].x and landmark[4].x < landmark[10].x) \
        and (landmark[5].y < landmark[0].y and landmark[17].y < landmark[0].y):
        return "N" # avoid E confusion AND thumb between correct digits AND hand oriented upwards (avoid G/N/M confusion)
    elif landmark[4].y < landmark[16].y \
        and (analyzeFinger(1, Finger.BENT_OR_PALM, landmark)) \
        and (analyzeFinger(2, Finger.BENT_OR_PALM, landmark)) \
        and (analyzeFinger(3, Finger.BENT_OR_PALM, landmark)) \
        and (analyzeFinger(4, Finger.BENT_OR_PALM, landmark)) \
        and (right_hand and landmark[14].x < landmark[4].x  or \
              left_hand and landmark[4].x  < landmark[14].x) \
        and (landmark[5].y < landmark[0].y and landmark[17].y < landmark[0].y):
        return "M" # avoid E confusion AND thumb between correct digits AND hand oriented upwards (avoid G/N/M confusion)
    elif analyzeFinger(1, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(2, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(3, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(4, Finger.STRAIGHT_UP, landmark) \
        and ((right_hand and landmark[4].x > landmark[8].x) or (left_hand and landmark[4].x < landmark[8].x)): # thumb tip crosses index finger
        return "B"
    # C 
    elif (right_hand \
        and analyzeFinger(1, Finger.DOWN_OR_LEFT, landmark) \
        and analyzeFinger(2, Finger.DOWN_OR_LEFT, landmark) \
        and analyzeFinger(3, Finger.DOWN_OR_LEFT, landmark) \
        and analyzeFinger(4, Finger.DOWN_OR_LEFT, landmark) \
        and analyzeFinger(0, Finger.UP_OR_LEFT, landmark) \
        and landmark[20].x < landmark[5].x \
        and distance(landmark[8], landmark[4]) > distance(landmark[4], landmark[3]) \
        and landmark[8].y < landmark[4].y and landmark[20].y < landmark[4].y) or \
        (left_hand \
        and analyzeFinger(1, Finger.DOWN_OR_RIGHT, landmark) \
        and analyzeFinger(2, Finger.DOWN_OR_RIGHT, landmark) \
        and analyzeFinger(3, Finger.DOWN_OR_RIGHT, landmark) \
        and analyzeFinger(4, Finger.DOWN_OR_RIGHT, landmark) \
        and analyzeFinger(0, Finger.UP_OR_RIGHT, landmark) \
        and landmark[20].x > landmark[5].x \
        and distance(landmark[8], landmark[4]) > distance(landmark[4], landmark[3]) \
        and landmark[8].y < landmark[4].y and landmark[20].y < landmark[4].y ): # TODO: distinguish from "O"
        return "C" # very much turned to the side AND large gap between index tip and thumb tip AND all finger tips above thumb tip
    elif analyzeFinger(1, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(2, Finger.PALM, landmark) \
        and analyzeFinger(3, Finger.PALM, landmark) \
        and analyzeFinger(4, Finger.PALM, landmark) \
        and (right_hand and landmark[4].x > landmark[8].x or left_hand and landmark[4].x < landmark[8].x): # thumb tip crosses index finger # LEFT D confused with I
        return "D"
    elif landmark[4].y > landmark[16].y \
        and (analyzeFinger(1, Finger.BENT, landmark) or analyzeFinger(1, Finger.PALM, landmark)) \
        and (analyzeFinger(2, Finger.BENT, landmark) or analyzeFinger(2, Finger.PALM, landmark)) \
        and (analyzeFinger(3, Finger.BENT, landmark) or analyzeFinger(3, Finger.PALM, landmark)) \
        and (analyzeFinger(4, Finger.BENT, landmark) or analyzeFinger(4, Finger.PALM, landmark)) \
        and (right_hand and analyzeFinger(0, Finger.RIGHT, landmark) or left_hand and analyzeFinger(0, Finger.LEFT, landmark)) \
        and (right_hand and landmark[12].x > landmark[5].x and landmark[12].x < landmark[13].x or left_hand and landmark[12].x < landmark[5].x and landmark[12].x > landmark[13].x) \
        and (right_hand and landmark[4].x > landmark[12].x or left_hand and landmark[4].x < landmark[12].x):
        return "E" # thumb is below fingertips (avoid T/N/M confusion) AND thumb is correct direction AND palm facing forward (not turned to *either* side) AND thumb far in towards the palm
    elif analyzeFinger(2, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(3, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(4, Finger.STRAIGHT_UP, landmark) \
        and not analyzeFinger(1, Finger.STRAIGHT, landmark): # no condition on thumb, too restrictive
        return "F" # TODO: add parallel condition on fingers 2/3/4? closeness of 0/1? 
    elif analyzeFinger(1, Finger.STRAIGHT_LEFT_OR_RIGHT, landmark) \
        and not analyzeFinger(2, Finger.STRAIGHT, landmark) \
        and not analyzeFinger(3, Finger.STRAIGHT, landmark) \
        and not analyzeFinger(4, Finger.STRAIGHT, landmark) \
        and landmark[4].y > landmark[8].y: # avoid "gun" gesture recognized as "G"
        return "G"
    elif analyzeFinger(1, Finger.STRAIGHT_LEFT_OR_RIGHT, landmark) \
        and analyzeFinger(2, Finger.STRAIGHT_LEFT_OR_RIGHT, landmark) \
        and not analyzeFinger(3, Finger.STRAIGHT, landmark) \
        and not analyzeFinger(4, Finger.STRAIGHT, landmark) \
        and landmark[4].y > landmark[8].y: # avoid "gun" gesture recognized as "H"
        return "H"                                        # TODO: small angle between fingers, similar to U vs V.... write a "angle_approx" function ????
    elif analyzeFinger(1, Finger.PALM, landmark) \
        and analyzeFinger(2, Finger.PALM, landmark) \
        and analyzeFinger(3, Finger.PALM, landmark) \
        and analyzeFinger(4, Finger.STRAIGHT, landmark) \
        and ((right_hand and landmark[8].x < landmark[4].x and landmark[4].x < landmark[20].x) or (left_hand and landmark[20].x < landmark[4].x and landmark[4].x < landmark[8].x)):
        return "I" # thump tip between index and pinky tip
    # K
    elif (right_hand and landmark[5].x < landmark[4].x and landmark[4].x < landmark[9].x or \
           left_hand and landmark[9].x < landmark[4].x and landmark[4].x < landmark[5].x) \
        and landmark[4].y < landmark[5].y \
        and analyzeFinger(1, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(2, Finger.STRAIGHT_UP, landmark) \
        and (analyzeFinger(3, Finger.BENT, landmark) or analyzeFinger(3, Finger.PALM, landmark))\
        and (analyzeFinger(4, Finger.BENT, landmark) or analyzeFinger(4, Finger.PALM, landmark)):
        return "K"
    # L
    elif (right_hand and analyzeFinger(0, Finger.LEFT, landmark) or left_hand and analyzeFinger(0, Finger.RIGHT, landmark)) \
        and analyzeFinger(1, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(2, Finger.PALM, landmark) \
        and analyzeFinger(3, Finger.PALM, landmark) \
        and analyzeFinger(4, Finger.PALM, landmark):
        return "L"
    # above: M, N
    # O
    elif (right_hand \
        and analyzeFinger(1, Finger.DOWN_OR_LEFT, landmark) \
        and analyzeFinger(2, Finger.DOWN_OR_LEFT, landmark) \
        and analyzeFinger(3, Finger.DOWN_OR_LEFT, landmark) \
        and analyzeFinger(4, Finger.DOWN_OR_LEFT, landmark) \
        and analyzeFinger(0, Finger.UP_OR_LEFT, landmark) \
        and landmark[20].x < landmark[5].x \
        and distance(landmark[8], landmark[4]) < distance(landmark[4], landmark[3])) or \
        (left_hand \
        and analyzeFinger(1, Finger.DOWN_OR_RIGHT, landmark) \
        and analyzeFinger(2, Finger.DOWN_OR_RIGHT, landmark) \
        and analyzeFinger(3, Finger.DOWN_OR_RIGHT, landmark) \
        and analyzeFinger(4, Finger.DOWN_OR_RIGHT, landmark) \
        and analyzeFinger(0, Finger.UP_OR_RIGHT, landmark) \
        and landmark[20].x > landmark[5].x \
        and distance(landmark[8], landmark[4]) < distance(landmark[4], landmark[3]) ): # TODO: distinguish from "O"
        return "O" # very much turned to the side AND *small* gap between index tip and thumb tip
    # P
    # Q
    # R
    # above: S, T
    # U
    elif (right_hand and landmark[4].x > landmark[9].x or left_hand and landmark[4].x < landmark[9].x) \
        and analyzeFinger(1, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(2, Finger.STRAIGHT_UP, landmark) \
        and abs( analyzeFinger(1, Finger.ANGLE, landmark) - analyzeFinger(2, Finger.ANGLE, landmark) ) < math.radians(10) \
        and (analyzeFinger(3, Finger.BENT, landmark) or analyzeFinger(3, Finger.PALM, landmark))\
        and (analyzeFinger(4, Finger.BENT, landmark) or analyzeFinger(4, Finger.PALM, landmark)):
        return "U" # TODO (here and "K": add angle between U,V)
    # V
    elif (right_hand and landmark[4].x > landmark[9].x or left_hand and landmark[4].x < landmark[9].x) \
        and analyzeFinger(1, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(2, Finger.STRAIGHT_UP, landmark) \
        and abs( analyzeFinger(1, Finger.ANGLE, landmark) - analyzeFinger(2, Finger.ANGLE, landmark) ) >= math.radians(10) \
        and (analyzeFinger(3, Finger.BENT, landmark) or analyzeFinger(3, Finger.PALM, landmark))\
        and (analyzeFinger(4, Finger.BENT, landmark) or analyzeFinger(4, Finger.PALM, landmark)):
        return "V" # TODO (here and "K": add angle between U,V)
    # W
    elif (right_hand and landmark[4].x > landmark[9].x or left_hand and landmark[4].x < landmark[9].x) \
        and analyzeFinger(1, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(2, Finger.STRAIGHT_UP, landmark) \
        and analyzeFinger(3, Finger.STRAIGHT_UP, landmark) \
        and (analyzeFinger(4, Finger.BENT, landmark) or analyzeFinger(4, Finger.PALM, landmark)):
        return "W" 
    # Y
    elif analyzeFinger(0, Finger.STRAIGHT, landmark) \
        and analyzeFinger(1, Finger.PALM, landmark) \
        and analyzeFinger(2, Finger.PALM, landmark) \
        and analyzeFinger(3, Finger.PALM, landmark) \
        and analyzeFinger(4, Finger.STRAIGHT, landmark):
        return "Y"
    # Z
    elif analyzeFinger(1, Finger.STRAIGHT_DOWN, landmark) \
        and analyzeFinger(2, Finger.BENT_OR_PALM, landmark) \
        and analyzeFinger(3, Finger.BENT_OR_PALM, landmark) \
        and analyzeFinger(4, Finger.BENT_OR_PALM, landmark) \
        and landmark[4].y > landmark[0].y:
        return "Z (or Q?)" # index finger tip below wrist - overall hand orientation is down ---- TODO: fix confusion with E, add another condition to E for "upright"
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
        # summary = analyzeFinger(1, Finger.HAND_FRONT, landmark)
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

