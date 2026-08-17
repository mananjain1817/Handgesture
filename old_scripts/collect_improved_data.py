import cv2
import mediapipe as mp
import csv
import os
import time

# ==========================================
# SETTINGS
# ==========================================

SAMPLES_PER_GESTURE = 300

GESTURES = {
    "0": "Fist",
    "2": "Thumbs Up",
    "5": "Pointing"
}

CSV_FILE = "dataset/gesture_data.csv"

# Capture every N frames
CAPTURE_INTERVAL = 5

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

# ==========================================
# CSV
# ==========================================

csv_file = open(
    CSV_FILE,
    mode="a",
    newline=""
)

writer = csv.writer(csv_file)

# ==========================================
# SELECT GESTURE
# ==========================================

print("\n==============================")
print("IMPROVED DATA COLLECTION")
print("==============================\n")

for key, gesture in GESTURES.items():
    print(f"{key} → {gesture}")

choice = input("\nEnter gesture number: ")

if choice not in GESTURES:
    print("Invalid gesture.")
    csv_file.close()
    cap.release()
    exit()

gesture_name = GESTURES[choice]
label = int(choice)

print(f"\nSelected gesture: {gesture_name}")
print(f"New samples: {SAMPLES_PER_GESTURE}")

input("\nPress ENTER when ready...")

# ==========================================
# COUNTDOWN
# ==========================================

print("\nGet your hand ready.")

for i in range(3, 0, -1):
    print(i)
    time.sleep(1)

print("START!")

# ==========================================
# COLLECTION
# ==========================================

count = 0
frame_count = 0

while count < SAMPLES_PER_GESTURE:

    ret, frame = cap.read()

    if not ret:
        print("Camera error.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    frame_count += 1

    if (
        results.multi_hand_landmarks
        and frame_count % CAPTURE_INTERVAL == 0
    ):

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

        # ------------------------------
        # Save
        # ------------------------------

        writer.writerow(
            landmarks + [label]
        )

        count += 1

        # ------------------------------
        # Draw
        # ------------------------------

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    # ==================================
    # DISPLAY
    # ==================================

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
        f"New samples: {count}/{SAMPLES_PER_GESTURE}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Move hand: position / angle / distance",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Improved Data Collection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# CLEANUP
# ==========================================

csv_file.close()
cap.release()
cv2.destroyAllWindows()
hands.close()

print("\n==============================")
print("COLLECTION COMPLETE")
print("==============================")

print(f"Gesture: {gesture_name}")
print(f"New samples: {count}")
print(f"Saved to: {CSV_FILE}")