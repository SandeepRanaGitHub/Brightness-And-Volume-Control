import cv2
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
import threading
import mediapipe as mp
import speech_recognition as sr

# Voice Recognition setup
r = sr.Recognizer()

# Hand Detection Model
class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        rlmlist = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) == 2:
                rlmlist.append('both')
            elif len(self.results.multi_hand_landmarks) == 1:
                rlmlist.append(self.results.multi_handedness[0].classification[0].label)

            for n in self.results.multi_hand_landmarks:
                lmList = []
                myHand = n
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                rlmlist.append(lmList)

        return rlmlist

# Initialize audio and brightness control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVolume = volRange[0]
maxVolume = volRange[1]
minBrightness = 0
maxBrightness = 100

# Functions to set volume and brightness
def setVolume(dist):
    vol = np.interp(int(dist), [35, 215], [minVolume, maxVolume])
    volume.SetMasterVolumeLevel(vol, None)

def setBrightness(dist):
    brightness = np.interp(int(dist), [35, 230], [minBrightness, maxBrightness])
    sbc.set_brightness(int(brightness))

# Voice control functions
def adjust_brightness_from_voice(text):
    if "brighter" in text:
        sbc.set_brightness('+10')
    elif "dimmer" in text:
        sbc.set_brightness('-10')

def adjust_volume_from_voice(text):
    if "increase volume" in text:
        volume.SetMasterVolumeLevelScalar(min(1.0, volume.GetMasterVolumeLevelScalar() + 0.1), None)
    elif "decrease volume" in text:
        volume.SetMasterVolumeLevelScalar(max(0.0, volume.GetMasterVolumeLevelScalar() - 0.1), None)

# Voice control loop
def voice_control():
    with sr.Microphone() as source:
        while True:
            print("Say something!")
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio, language="en-US")
                print("Voice command:", text)
                if "brightness" in text:
                    adjust_brightness_from_voice(text)
                elif "volume" in text:
                    adjust_volume_from_voice(text)
            except sr.UnknownValueError:
                print("Couldn't understand audio.")
            except sr.RequestError as e:
                print(f"Error with Google API: {e}")

# Hand detection loop
def hand_detection_loop():
    vidObj = cv2.VideoCapture(0)
    vidObj.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vidObj.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    handlmsObj = handDetector(detectionCon=0.7)
    camera_active = True

    while True:
        if camera_active:
            _, frame = vidObj.read()
            frame = cv2.flip(frame, 1)
            frame = handlmsObj.findHands(frame)
            lndmrks = handlmsObj.findPosition(frame, draw=False)
            if lndmrks:
                xr1, yr1 = lndmrks[1][4][1], lndmrks[1][4][2]
                xr2, yr2 = lndmrks[1][8][1], lndmrks[1][8][2]
                dist = math.hypot(xr2 - xr1, yr2 - yr1)
                if lndmrks[0] == 'Left':
                    setBrightness(dist)
                elif lndmrks[0] == 'Right':
                    setVolume(dist)

            cv2.imshow("Hand Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                camera_active = False
                vidObj.release()
                cv2.destroyWindow("Hand Control")
                print("Camera closed. Voice control still active.")
                break

# Start threads for hand detection and voice control
voice_thread = threading.Thread(target=voice_control, daemon=True)
voice_thread.start()
hand_detection_loop()

# Keep the program running so voice thread stays active
voice_thread.join()
