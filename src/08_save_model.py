from pathlib import Path
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "processed_train.csv")

X = df["clean_plot"]
y = df["genre"]

model_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            max_features=15000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LinearSVC(
            C=0.5,
            class_weight="balanced",
            random_state=42
        )
    )
])

print("Training final model...")

model_pipeline.fit(X, y)

model_path = MODEL_DIR / "cine_sense_model.pkl"

joblib.dump(
    model_pipeline,
    model_path,
    compress=3
)

print(f"✓ Model saved successfully: {model_path}")

size_mb = model_path.stat().st_size / (1024 * 1024)

print(f"✓ Model size: {size_mb:.2f} MB")