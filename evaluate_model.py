import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "models/gesture_model.pkl"

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================
# LOAD INDEPENDENT TEST DATA
# ==========================================

TEST_PATH = "real_test/test_data.csv"

df = pd.read_csv(TEST_PATH)

print("Test dataset shape:", df.shape)

# ==========================================
# SEPARATE FEATURES AND LABEL
# ==========================================

X_test = df.drop("label", axis=1)
y_test = df["label"]

# ==========================================
# PREDICTION
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==============================")
print("INDEPENDENT TEST RESULTS")
print("==============================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

class_names = [
    "Fist",
    "Open Palm",
    "Thumbs Up",
    "Thumbs Down",
    "Victory",
    "Pointing"
]

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=class_names
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)