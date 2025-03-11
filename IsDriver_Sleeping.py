import cv2
import numpy as np
import dlib
import threading
import playsound
from imutils import face_utils

# Load models
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure correct path

# Counters & status
sleep, drowsy, active = 0, 0, 0
status = ""
color = (0, 0, 0)

# Function to play sounds
def play_sound(sound_file):
    threading.Thread(target=playsound.playsound, args=(sound_file,), daemon=True).start()

# Compute Euclidean distance
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Eye aspect ratio (EAR) calculation
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2  # Eyes open
    elif 0.21 <= ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Eyes closed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        if landmarks is None:
            continue

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Detect drowsiness
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep == 7:  # Play sound only when sleep is detected
                status = "Alert SLEEPING "
                color = (255, 0, 0)
                play_sound("Sound/sleeping_alaram.mp3")  # Play sleeping alert

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy == 7:  # Play sound only when drowsy is detected
                status = "Feeling Drowsy"
                color = (0, 0, 255)
                play_sound("Sound/drowsy_alaram.mp3")  # Play drowsy alert

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = " Active "
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
