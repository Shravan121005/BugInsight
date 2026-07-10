import joblib
from pathlib import Path

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# ==========================================================
# Load Data
# ==========================================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    _,
    _,
    _,
    _,
) = joblib.load(MODEL_DIR / "dataset.pkl")

# ==========================================================
# Encode Labels
# ==========================================================

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")

# ==========================================================
# Train Model
# ==========================================================

model = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
)

model.fit(X_train, y_train)

# ==========================================================
# Prediction
# ==========================================================

predictions = model.predict(X_test)

# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
)

print("=" * 60)
print("Severity Prediction Results")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")
print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
    )
)

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, predictions))

# ==========================================================
# Save Model
# ==========================================================

joblib.dump(
    model,
    MODEL_DIR / "severity_model.pkl",
)

print("\nSaved:")
print("✔ severity_model.pkl")
print("✔ label_encoder.pkl")