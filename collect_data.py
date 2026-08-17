import cv2
import mediapipe as mp
import csv
import os
import time

# ==============================
# SETTINGS
# ==============================

SAMPLES_PER_GESTURE = 500

GESTURES = {
    "0": "Fist",
    "1": "Open Palm",
    "2": "Thumbs Up",
    "3": "Thumbs Down",
    "4": "Victory",
    "5": "Pointing"
}

# ==============================
# CREATE DATASET FOLDER
# ==============================

os.makedirs("dataset", exist_ok=True)

CSV_FILE = "dataset/gesture_data.csv"

# ==============================
# MEDIAPIPE SETUP
# ==============================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==============================
# CREATE CSV FILE
# ==============================

file_exists = os.path.exists(CSV_FILE)

csv_file = open(
    CSV_FILE,
    mode="a",
    newline=""
)

writer = csv.writer(csv_file)

if not file_exists:
    header = []

    for i in range(21):
        header.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}"
        ])

    header.append("label")
    writer.writerow(header)

# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

# ==============================
# SELECT GESTURE
# ==============================

print("\n==============================")
print("HAND GESTURE DATA COLLECTION")
print("==============================\n")

for key, gesture in GESTURES.items():
    print(f"{key} → {gesture}")

print("\nEnter the number of the gesture you want to collect.")

choice = input("\nEnter gesture number: ")

if choice not in GESTURES:
    print("Invalid gesture number.")
    cap.release()
    csv_file.close()
    exit()

gesture_name = GESTURES[choice]
label = int(choice)

print(f"\nSelected gesture: {gesture_name}")
print(f"Samples to collect: {SAMPLES_PER_GESTURE}")

input("\nPress ENTER when you are ready...")

print("\nGet your hand into position.")

# ==============================
# COUNTDOWN
# ==============================

for i in range(3, 0, -1):
    print(i)
    time.sleep(1)

print("START!")

# ==============================
# DATA COLLECTION
# ==============================

count = 0

while count < SAMPLES_PER_GESTURE:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
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
        # Get wrist coordinates
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
        # Save to CSV
        # ------------------------------

        writer.writerow(
            landmarks + [label]
        )

        count += 1

        # ------------------------------
        # Draw hand
        # ------------------------------

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    # ------------------------------
    # Display information
    # ------------------------------

    cv2.putText(
        frame,
        f"Gesture: {gesture_name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {count}/{SAMPLES_PER_GESTURE}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Gesture Data Collection",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================

csv_file.close()
cap.release()
cv2.destroyAllWindows()
hands.close()

print("\n==============================")
print("DATA COLLECTION COMPLETE")
print("==============================")
print(f"Gesture: {gesture_name}")
print(f"Samples collected: {count}")
print(f"Saved to: {CSV_FILE}")