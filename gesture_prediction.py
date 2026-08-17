import cv2
import mediapipe as mp
import numpy as np
import joblib

# ==========================================
# LOAD TRAINED MODEL
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
    print("ERROR: Could not open camera.")
    exit()

print("Real-time gesture recognition started.")
print("Press Q to quit.")

# ==========================================
# REAL-TIME PREDICTION
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # ------------------------------
        # Wrist coordinates
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

        # ------------------------------
        # Convert to NumPy array
        # ------------------------------

        features = np.array(
            landmarks
        ).reshape(1, -1)

        # ------------------------------
        # Predict gesture
        # ------------------------------

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = np.max(probabilities) * 100

        gesture_name = GESTURES[prediction]

        # ------------------------------
        # Draw landmarks
        # ------------------------------

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # ------------------------------
        # Display prediction
        # ------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (380, 100),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Gesture: {gesture_name}",
            (20, 45),
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

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # ------------------------------
    # Display
    # ------------------------------

    cv2.imshow(
        "AI Hand Gesture Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()