from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Base directory setup
BASE_DIR = Path(__file__).resolve().parents[2]


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100


def calculate_r2(y_true, y_pred):
    # Custom R2 calculation to avoid dependency warning issues on single rows
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def train_and_compare():
    print("Memulai Phase 5: Perbaikan dan Perbandingan Model...")

    # Path data
    train_path = BASE_DIR / "data/features/train.csv"
    val_path = BASE_DIR / "data/features/val.csv"
    model_output_path = BASE_DIR / "models/model.pkl"
    features_output_path = BASE_DIR / "models/features.pkl"

    # Muat data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Gabungkan train dan val untuk cross validation temporal
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    feature_cols = ['lag_1', 'lag_2', 'rolling_mean_3', 'growth_rate']
    target_col = 'target'

    # Dataset split
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_full = full_train_df[feature_cols]
    y_full = full_train_df[target_col]

    # Inisialisasi CV temporal
    tscv = TimeSeriesSplit(n_splits=3)

    results = []

    # ==========================================
    # 1. Model Baseline (Naive Forecast)
    # Naive forecast memprediksi Target(t) = NetCash(t+1) adalah net_cash(t), yang bernilai lag_1
    # ==========================================
    y_pred_naive_val = X_val['lag_1']
    results.append({
        'Model': 'Baseline (Naive)',
        'MAE': mean_absolute_error(y_val, y_pred_naive_val),
        'RMSE': root_mean_squared_error(y_val, y_pred_naive_val),
        'MAPE': calculate_mape(y_val.values, y_pred_naive_val.values),
        'R2': calculate_r2(y_val.values, y_pred_naive_val.values),
        'Object': 'naive'
    })

    # ==========================================
    # 2. Linear Regression
    # ==========================================
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr_val = lr.predict(X_val)
    results.append({
        'Model': 'Linear Regression',
        'MAE': mean_absolute_error(y_val, y_pred_lr_val),
        'RMSE': root_mean_squared_error(y_val, y_pred_lr_val),
        'MAPE': calculate_mape(y_val.values, y_pred_lr_val),
        'R2': calculate_r2(y_val.values, y_pred_lr_val),
        'Object': lr
    })

    # ==========================================
    # 3. Random Forest Regressor
    # ==========================================
    rf = RandomForestRegressor(random_state=42)
    rf_params = {
        'max_depth': [2, 3, 4],
        'n_estimators': [10, 30, 50]
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    rf_grid.fit(X_full, y_full)
    best_rf = rf_grid.best_estimator_
    y_pred_rf_val = best_rf.predict(X_val)
    results.append({
        'Model': 'Random Forest',
        'MAE': mean_absolute_error(y_val, y_pred_rf_val),
        'RMSE': root_mean_squared_error(y_val, y_pred_rf_val),
        'MAPE': calculate_mape(y_val.values, y_pred_rf_val),
        'R2': calculate_r2(y_val.values, y_pred_rf_val),
        'Object': best_rf
    })

    # ==========================================
    # 4. XGBoost Regressor (Model Utama)
    # ==========================================
    xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
    xgb_params = {
        'max_depth': [2, 3, 4],
        'n_estimators': [10, 30, 50],
        'learning_rate': [0.03, 0.05, 0.1]
    }
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    xgb_grid.fit(X_full, y_full)
    best_xgb = xgb_grid.best_estimator_
    y_pred_xgb_val = best_xgb.predict(X_val)
    results.append({
        'Model': 'XGBoost (Utama)',
        'MAE': mean_absolute_error(y_val, y_pred_xgb_val),
        'RMSE': root_mean_squared_error(y_val, y_pred_xgb_val),
        'MAPE': calculate_mape(y_val.values, y_pred_xgb_val),
        'R2': calculate_r2(y_val.values, y_pred_xgb_val),
        'Object': best_xgb
    })

    # ==========================================
    # Ringkasan Perbandingan
    # ==========================================
    summary_df = pd.DataFrame(results)
    print("\n=== Tabel Perbandingan Model (Validation Set) ===")
    print(summary_df[['Model', 'MAE', 'RMSE', 'MAPE', 'R2']].to_string(index=False))

    # Kita pilih XGBoost sebagai model utama proyek sesuai ketentuan tugas
    best_model_info = summary_df[summary_df['Model'] == 'XGBoost (Utama)'].iloc[0]
    best_model = best_model_info['Object']

    # Simpan model XGBoost
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(best_model, f)

    with open(features_output_path, 'wb') as f:
        pickle.dump(feature_cols, f)

    print(f"\n[OK] Model XGBoost tersimpan di: {model_output_path}")
    print(f"[OK] Kolom fitur tersimpan di: {features_output_path}")


if __name__ == "__main__":
    train_and_compare()
