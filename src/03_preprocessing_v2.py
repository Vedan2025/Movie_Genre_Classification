from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

train_df = pd.read_csv(
    DATA_DIR / "train_data.txt",
    sep=" ::: ",
    engine="python",
    names=["id", "title", "genre", "plot"]
)

def clean_text(text):
    text = str(text).lower()

    # Remove URLs only
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

print("Cleaning text...")

train_df["clean_plot"] = train_df["plot"].apply(clean_text)

train_df.to_csv(
    DATA_DIR / "processed_train_v2.csv",
    index=False
)

print("✓ Saved: processed_train_v2.csv")