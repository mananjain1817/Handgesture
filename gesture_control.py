import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import time

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL


# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "models/gesture_model.pkl"

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# GESTURE LABELS
# ==========================================

GESTURES = {
    0: "Fist",
    1: "Open Palm",
    2: "Thumbs Up",
    3: "Thumbs Down",
    4: "Victory",
    5: "Pointing"
}


# ==========================================
# WINDOWS VOLUME CONTROL
# ==========================================

devices = AudioUtilities.GetSpeakers()
interface = devices.EndpointVolume
volume = interface




# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")
    exit()

print("Gesture control started.")
print("Press Q to quit.")


# ==========================================
# ACTION CONTROL
# ==========================================

last_action = None
last_action_time = 0

ACTION_COOLDOWN = 1.5


def perform_action(gesture):

    global last_action
    global last_action_time

    current_time = time.time()

    # Prevent repeated actions
    if (
        gesture == last_action
        and current_time - last_action_time < ACTION_COOLDOWN
    ):
        return None

    # ------------------------------
    # THUMBS UP
    # ------------------------------

    if gesture == 2:

        current_volume = volume.GetMasterVolumeLevelScalar()

        new_volume = min(
            current_volume + 0.10,
            1.0
        )

        volume.SetMasterVolumeLevelScalar(
            new_volume,
            None
        )

        action = "Volume Up"

    # ------------------------------
    # THUMBS DOWN
    # ------------------------------

    elif gesture == 3:

        current_volume = volume.GetMasterVolumeLevelScalar()

        new_volume = max(
            current_volume - 0.10,
            0.0
        )

        volume.SetMasterVolumeLevelScalar(
            new_volume,
            None
        )

        action = "Volume Down"

    # ------------------------------
    # VICTORY
    # ------------------------------

    elif gesture == 4:

        pyautogui.press("space")

        action = "Play / Pause"

    # ------------------------------
    # POINTING
    # ------------------------------

    elif gesture == 5:

        pyautogui.press("right")

        action = "Next"

    # ------------------------------
    # FIST
    # ------------------------------

    elif gesture == 0:

        pyautogui.screenshot(
            "gesture_screenshot.png"
        )

        action = "Screenshot Saved"

    # ------------------------------
    # OPEN PALM
    # ------------------------------

    elif gesture == 1:

        action = "Neutral"

    else:

        action = None

    if action is not None:

        last_action = gesture
        last_action_time = current_time

    return action


# ==========================================
# MAIN LOOP
# ==========================================

current_action = "None"

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(
        frame,
        1
    )

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(
        rgb_frame
    )

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # ------------------------------
        # Wrist
        # ------------------------------

        wrist = hand_landmarks.landmark[0]

        wrist_x = wrist.x
        wrist_y = wrist.y
        wrist_z = wrist.z

        landmarks = []

        # ------------------------------
        # Normalize landmarks
        # ------------------------------

        for landmark in hand_landmarks.landmark:

            x = landmark.x - wrist_x
            y = landmark.y - wrist_y
            z = landmark.z - wrist_z

            landmarks.extend([
                x,
                y,
                z
            ])

        features = np.array(
            landmarks
        ).reshape(1, -1)

        # ------------------------------
        # Prediction
        # ------------------------------

        prediction = model.predict(
            features
        )[0]

        gesture_name = GESTURES[prediction]

        # ------------------------------
        # Confidence
        # ------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                features
            )[0]

            confidence = (
                np.max(probabilities) * 100
            )

        else:

            confidence = 0

        # ------------------------------
        # Execute action
        # ------------------------------

        if confidence >= 70:

            action = perform_action(
                prediction
            )

            if action is not None:
                current_action = action

        # ------------------------------
        # Draw landmarks
        # ------------------------------

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # ------------------------------
        # Display gesture
        # ------------------------------

        cv2.putText(
            frame,
            f"Gesture: {gesture_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Action: {current_action}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "AI Gesture Computer Control",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()
hands.close()

print("Gesture control stopped.")