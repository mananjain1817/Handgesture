import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATASET_PATH = "dataset/gesture_data.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND LABEL
# ==========================================

X = df.drop("label", axis=1)
y = df["label"]

print("\nFeatures:", X.shape)
print("Labels:", y.shape)


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. RANDOM FOREST
# ==========================================

print("\n==============================")
print("Training Random Forest...")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("\nRandom Forest Accuracy:")
print(f"{rf_accuracy * 100:.2f}%")


print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=[
            "Fist",
            "Open Palm",
            "Thumbs Up",
            "Thumbs Down",
            "Victory",
            "Pointing"
        ]
    )
)


print("\nRandom Forest Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)


# ==========================================
# 5. SVM
# ==========================================

print("\n==============================")
print("Training SVM...")
print("==============================")

svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True
)

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

print("\nSVM Accuracy:")
print(f"{svm_accuracy * 100:.2f}%")


print("\nSVM Classification Report:")
print(
    classification_report(
        y_test,
        svm_predictions,
        target_names=[
            "Fist",
            "Open Palm",
            "Thumbs Up",
            "Thumbs Down",
            "Victory",
            "Pointing"
        ]
    )
)


print("\nSVM Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        svm_predictions
    )
)


# ==========================================
# 6. COMPARE MODELS
# ==========================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(f"Random Forest: {rf_accuracy * 100:.2f}%")
print(f"SVM:           {svm_accuracy * 100:.2f}%")


# ==========================================
# 7. SELECT BEST MODEL
# ==========================================

if rf_accuracy >= svm_accuracy:

    best_model = rf_model
    best_model_name = "Random Forest"

else:

    best_model = svm_model
    best_model_name = "SVM"


print(f"\nBest Model: {best_model_name}")


# ==========================================
# 8. CREATE MODELS FOLDER
# ==========================================

os.makedirs("models", exist_ok=True)


# ==========================================
# 9. SAVE BEST MODEL
# ==========================================

MODEL_PATH = "models/gesture_model.pkl"

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Saved to:", MODEL_PATH)