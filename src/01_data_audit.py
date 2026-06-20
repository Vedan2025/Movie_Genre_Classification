from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_FILE = DATA_DIR / "train_data.txt"
TEST_FILE = DATA_DIR / "test_data.txt"

print("=" * 60)
print("MOVIE GENRE CLASSIFICATION - DATA AUDIT")
print("=" * 60)

# Check files exist
for file in [TRAIN_FILE, TEST_FILE]:
    print(f"\nChecking: {file.name}")

    if file.exists():
        print("✓ Found")
    else:
        print("✗ Missing")

print("\nLoading training data...")

train_df = pd.read_csv(
    TRAIN_FILE,
    sep=" ::: ",
    engine="python",
    names=["id", "title", "genre", "plot"]
)

print("✓ Training data loaded")

print("\nLoading test data...")

test_df = pd.read_csv(
    TEST_FILE,
    sep=" ::: ",
    engine="python",
    names=["id", "title", "plot"]
)

print("✓ Test data loaded")

print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Training samples : {len(train_df):,}")
print(f"Test samples     : {len(test_df):,}")
print(f"Unique genres    : {train_df['genre'].nunique()}")

print("\nColumns:")
print(train_df.columns.tolist())

print("\nMissing values:")
print(train_df.isnull().sum())

print("\nDuplicate plots:")
print(train_df['plot'].duplicated().sum())

print("\nTop 10 genres:")
print(train_df['genre'].value_counts().head(10))

print("\nPlot length statistics:")

plot_lengths = train_df["plot"].str.split().str.len()

print(plot_lengths.describe())

print("\nSample records:")
print(train_df.head(3))

print("\n✓ Data audit completed successfully.")