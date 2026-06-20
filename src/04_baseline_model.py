from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

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
    max_features=50000,
    ngram_range=(1, 2),
    min_df=3
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

print("Training model...")

model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_valid_tfidf)

macro_f1 = f1_score(
    y_valid,
    predictions,
    average="macro"
)

print(f"\nMacro F1 Score: {macro_f1:.4f}\n")

print(classification_report(y_valid, predictions))