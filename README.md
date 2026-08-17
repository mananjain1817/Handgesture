# Hand Gesture Recognition and Computer Control

This project is a real-time hand gesture recognition system developed using Python, OpenCV, MediaPipe and machine learning.

The webcam captures the hand, MediaPipe detects 21 hand landmarks, and an SVM model classifies the gesture. The recognized gesture is then mapped to an action on the laptop.

## Gestures

| Gesture | Action |
|---|---|
| Fist | Screenshot |
| Open Palm | Neutral |
| Thumbs Up | Volume Up |
| Thumbs Down | Volume Down |
| Victory | Play/Pause |
| Pointing | Right Arrow |

## Technologies

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- PyAutoGUI
- PyCaw

## Dataset

- 4,500 training samples
- 600 independent test samples
- 6 gesture classes
- 21 MediaPipe landmarks
- 63 features

## Results

Random Forest: 99.22% internal accuracy

SVM: 99.89% internal accuracy

Independent test accuracy: **91.67%**

The independent test was performed using separately collected samples that were not used for training.

## How to Run

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

Run the application:

```powershell
python gesture_control.py
```

The webcam will open and the system will start recognizing the gestures.

Press `Q` to close the application.

## Project Structure

```text
Handgesture/
│
├── dataset/
│   └── gesture_data.csv
│
├── models/
│   └── gesture_model.pkl
│
├── real_test/
│   └── test_data.csv
│
├── old_scripts/
│
├── collect_data.py
├── evaluate_model.py
├── gesture_control.py
├── gesture_prediction.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Limitations

- Performance can vary depending on lighting conditions.
- Camera quality can affect hand detection.
- Background conditions can affect recognition.
- Similar gestures can sometimes be confused.
- The current system supports one hand.
- The model currently recognizes six predefined gestures.
- Keyboard-based actions depend on the application currently in focus.

## Future Scope

- Add more gesture classes.
- Support two-hand gesture recognition.
- Collect a larger and more diverse dataset.
- Improve recognition of similar gestures.
- Allow users to create custom gestures.
- Add application-specific controls.
- Improve gesture smoothing.

## Conclusion

This project combines computer vision and machine learning to create a real-time contactless computer-control system.

MediaPipe is used for hand landmark detection and SVM is used for gesture classification. The recognized gestures are then mapped to different actions on the laptop.

The final model achieved an independent test accuracy of 91.67%.