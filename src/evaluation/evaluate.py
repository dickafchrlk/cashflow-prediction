from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Base directory setup
BASE_DIR = Path(__file__).resolve().parents[2]


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100


def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def evaluate():
    print("Memulai Phase 6: Evaluasi Model Bulanan/Mingguan...")

    test_path = BASE_DIR / "data/features/test.csv"
    model_path = BASE_DIR / "models/model.pkl"
    metrics_output_path = BASE_DIR / "reports/metrics.json"
    figures_dir = BASE_DIR / "reports/figures"

    # Muat data test
    test_df = pd.read_csv(test_path)

    # Muat model
    if not model_path.exists():
        print(f"ERROR: File model {model_path} tidak ditemukan!")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    feature_cols = ['lag_1', 'lag_2', 'rolling_mean_3', 'growth_rate']
    target_col = 'target'

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    weeks = test_df['week']

    # Prediksi
    y_pred = model.predict(X_test)

    # Hitung metrik
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = calculate_mape(y_test.values, y_pred)
    r2 = calculate_r2(y_test.values, y_pred)

    print("\nHasil Evaluasi Akhir (Test Set):")
    print(f"  - MAE:  Rp {mae:,.2f}")
    print(f"  - RMSE: Rp {rmse:,.2f}")
    print(f"  - MAPE: {mape:.2f}%")
    print(f"  - R²:   {r2:.4f}")

    # Simpan metrik ke JSON
    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2)
    }

    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\n[OK] Metrik evaluasi bulanan/mingguan disimpan di: {metrics_output_path}")

    # Plot Aktual vs Prediksi
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(weeks, y_test, marker='o', label='Aktual', color='blue', markersize=8)
    plt.plot(weeks, y_pred, marker='x', label='Prediksi XGBoost', color='red', linestyle='--', markersize=8)
    plt.title('Kurva Perbandingan Aktual vs Prediksi (Test Set)', fontsize=12)
    plt.xlabel('Minggu', fontsize=10)
    plt.ylabel('Nominal (Rupiah)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    plt.tight_layout()

    chart_path = figures_dir / "actual_vs_predicted.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[OK] Grafik visualisasi disimpan di: {chart_path}")

    # Plot Feature Importance
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(8, 5))
        plt.title('Tingkat Kepentingan Fitur (Feature Importance) - XGBoost', fontsize=12)
        plt.barh(range(len(feature_cols)), importances[indices][::-1], align='center', color='teal')
        plt.yticks(range(len(feature_cols)), [feature_cols[i] for i in indices][::-1])
        plt.xlabel('Skor Kepentingan', fontsize=10)
        plt.tight_layout()

        fi_chart_path = figures_dir / "feature_importance.png"
        plt.savefig(fi_chart_path, dpi=150)
        plt.close()
        print(f"[OK] Grafik feature importance disimpan di: {fi_chart_path}")
    except Exception as e:
        print(f"Bypass Feature Importance Plot: {e}")


if __name__ == "__main__":
    evaluate()
