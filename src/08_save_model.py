from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
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
            max_features=100000,
            ngram_range=(1, 2),
            min_df=3,
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

joblib.dump(
    model_pipeline,
    MODEL_DIR / "cine_sense_model.pkl"
)

print("✓ Model saved successfully")