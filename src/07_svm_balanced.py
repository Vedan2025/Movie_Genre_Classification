from pathlib import Path
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

df = pd.read_csv(DATA_DIR / "processed_train.csv")

X = df["clean_plot"]
y = df["genre"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True
)

print("Creating TF-IDF features...")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)

model = LinearSVC(
    C=0.5,
    class_weight="balanced",
    random_state=42
)

print("Training model...")

start = time.time()

model.fit(X_train_tfidf, y_train)

training_time = time.time() - start

predictions = model.predict(X_valid_tfidf)

accuracy = accuracy_score(y_valid, predictions)

macro_f1 = f1_score(
    y_valid,
    predictions,
    average="macro"
)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Accuracy      : {accuracy:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"Training Time : {training_time:.2f} seconds")

print("\n")

print(
    classification_report(
        y_valid,
        predictions,
        zero_division=0
    )
)