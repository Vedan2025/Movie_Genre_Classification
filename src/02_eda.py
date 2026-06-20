from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = PROJECT_ROOT / "images"

IMAGE_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(
    DATA_DIR / "train_data.txt",
    sep=" ::: ",
    engine="python",
    names=["id", "title", "genre", "plot"]
)

# Genre distribution
genre_counts = train_df["genre"].value_counts()

plt.figure(figsize=(12, 8))

sns.barplot(
    x=genre_counts.values,
    y=genre_counts.index,
)

plt.title("Movie Genre Distribution")
plt.xlabel("Number of Movies")
plt.ylabel("Genre")

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "genre_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Plot length analysis
train_df["plot_length"] = train_df["plot"].str.split().str.len()

plt.figure(figsize=(10, 6))

sns.histplot(
    train_df["plot_length"],
    bins=50,
    kde=True
)

plt.title("Distribution of Plot Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "plot_length_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nEDA completed.")