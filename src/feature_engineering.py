import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import joblib

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "processed_bugs.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(DATA_PATH)

# ==========================================================
# Features & Targets
# ==========================================================

X = df[
    [
        "text",
        "component_name",
        "product_name",
        "quantity_of_votes",
        "quantity_of_comments",
    ]
]

y_class = df["severity_category"]
y_reg = df["bug_fix_time"]

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_class_train, y_class_test = train_test_split(
    X,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class,
)

y_reg_train = y_reg.loc[X_train.index]
y_reg_test = y_reg.loc[X_test.index]

# ==========================================================
# Preprocessor
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(max_features=5000),
            "text",
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            ["component_name", "product_name"],
        ),
        (
            "num",
            "passthrough",
            ["quantity_of_votes", "quantity_of_comments"],
        ),
    ]
)

# ==========================================================
# Transform Data
# ==========================================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ==========================================================
# Save Objects
# ==========================================================

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.pkl",
)

joblib.dump(
    (
        X_train_processed,
        X_test_processed,
        y_class_train,
        y_class_test,
        y_reg_train,
        y_reg_test,
        X_train.index,
        X_test.index,
    ),
    MODEL_DIR / "dataset.pkl",
)

# ==========================================================
# Summary
# ==========================================================

print("=" * 60)
print("Feature Engineering Completed Successfully")
print("=" * 60)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")

print(f"Processed Train Shape : {X_train_processed.shape}")
print(f"Processed Test Shape  : {X_test_processed.shape}")

print("\nSaved Files")
print("-----------------------")
print("✔ preprocessor.pkl")
print("✔ dataset.pkl")