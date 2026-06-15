from pathlib import Path
import pandas as pd

# Base directory setup
BASE_DIR = Path(__file__).resolve().parents[2]


def split_data_monthly():
    print("Memulai pembagian data bulanan (Train-Val-Test Split)...")

    features_path = BASE_DIR / "data/features/features.csv"
    train_path = BASE_DIR / "data/features/train.csv"
    val_path = BASE_DIR / "data/features/val.csv"
    test_path = BASE_DIR / "data/features/test.csv"

    # Muat dataset fitur
    df = pd.read_csv(features_path)

    # Total baris adalah 6
    # 70% Train = 4 baris
    # 15% Validation = 1 baris
    # 15% Test = 1 baris
    train_df = df.iloc[:4].reset_index(drop=True)
    val_df = df.iloc[4:5].reset_index(drop=True)
    test_df = df.iloc[5:].reset_index(drop=True)

    # Ekspor ke file CSV
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[OK] Pembagian data selesai!")
    print(f"  - Train: {train_df.shape[0]} baris ({train_df['month'].min()} s/d {train_df['month'].max()})")
    print(f"  - Validation: {val_df.shape[0]} baris ({val_df['month'].iloc[0]})")
    print(f"  - Test: {test_df.shape[0]} baris ({test_df['month'].iloc[0]})")


if __name__ == "__main__":
    split_data_monthly()
