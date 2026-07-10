import joblib
from pathlib import Path

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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
    _,
    _,
    y_train,
    y_test,
    _,
    _,
) = joblib.load(MODEL_DIR / "dataset.pkl")

# ==========================================================
# Model
# ==========================================================

model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
)

# ==========================================================
# Train
# ==========================================================

model.fit(X_train, y_train)

# ==========================================================
# Predict
# ==========================================================

predictions = model.predict(X_test)

# ==========================================================
# Evaluation
# ==========================================================

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions,
) ** 0.5

r2 = r2_score(
    y_test,
    predictions,
)

print("=" * 60)
print("Resolution Time Prediction Results")
print("=" * 60)

print(f"MAE  : {mae:.2f} days")
print(f"RMSE : {rmse:.2f} days")
print(f"R²   : {r2:.4f}")

# ==========================================================
# Save Model
# ==========================================================

joblib.dump(
    model,
    MODEL_DIR / "resolution_model.pkl",
)

print("\nSaved:")
print("✔ resolution_model.pkl")