import cv2

# Open the default webcam
cap = cv2.VideoCapture(0)

# Check if camera opened
if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera opened successfully!")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Flip image horizontally so it behaves like a mirror
    frame = cv2.flip(frame, 1)

    # Display the camera
    cv2.imshow("Camera Test", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()