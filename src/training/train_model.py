from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Base directory setup
BASE_DIR = Path(__file__).resolve().parents[2]


def train_monthly_model():
    print("Memulai pelatihan model XGBoost Bulanan...")

    # Path file input dan output
    train_path = BASE_DIR / "data/features/train.csv"
    val_path = BASE_DIR / "data/features/val.csv"
    model_output_path = BASE_DIR / "models/model.pkl"
    features_output_path = BASE_DIR / "models/features.pkl"

    # Muat data train dan validation
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Tentukan fitur dan target sesuai ketentuan
    feature_cols = ['lag_1', 'lag_2', 'rolling_mean_3', 'growth_rate']
    target_col = 'target'

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    print(f"Fitur Latih: {feature_cols}")
    print(f"Target: {target_col}")

    # Latih XGBRegressor dengan hyperparameter konservatif karena ukuran data kecil
    model = XGBRegressor(
        n_estimators=15,
        max_depth=2,
        learning_rate=0.05,
        random_state=42,
        objective='reg:squarederror'
    )

    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Evaluasi sederhana pada data training (sanity check)
    y_pred_train = model.predict(X_train)
    train_rmse = root_mean_squared_error(y_train, y_pred_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)

    print("\nHasil Pelatihan (Data Latih):")
    print(f"  - RMSE: Rp {train_rmse:,.2f}")
    print(f"  - MAE:  Rp {train_mae:,.2f}")

    # Serialisasi Model dan Fitur menggunakan PICKLE (.pkl)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
        
    with open(features_output_path, 'wb') as f:
        pickle.dump(feature_cols, f)

    print(f"\n[OK] Model bulanan disimpan di: {model_output_path}")
    print(f"[OK] Daftar fitur bulanan disimpan di: {features_output_path}")


if __name__ == "__main__":
    train_monthly_model()
