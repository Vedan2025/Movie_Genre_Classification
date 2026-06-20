from pathlib import Path
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

print("=" * 60)
print("TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

# Load processed data
df = pd.read_csv(DATA_DIR / "processed_train_v2.csv")

X = df["clean_plot"]
y = df["genre"]

print(f"\nTotal samples: {len(df):,}")
print(f"Genres: {y.nunique()}")

# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    max_features=150000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)

print(f"Training matrix shape: {X_train_tfidf.shape}")

model = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    random_state=42
)

print("\nTraining model...")

start_time = time.time()

model.fit(X_train_tfidf, y_train)

training_time = time.time() - start_time

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

print("\nClassification Report:\n")

print(
    classification_report(
        y_valid,
        predictions,
        zero_division=0
    )
)