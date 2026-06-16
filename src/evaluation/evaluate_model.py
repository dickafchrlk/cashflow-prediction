from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Base directory setup
BASE_DIR = Path(__file__).resolve().parents[2]


def evaluate_weekly():
    print("Memulai evaluasi model mingguan pada dataset Testing...")

    # Path file input dan output
    test_path = BASE_DIR / "data/features/test.csv"
    model_path = BASE_DIR / "models/model.pkl"
    metrics_output_path = BASE_DIR / "reports/metrics.json"
    figures_dir = BASE_DIR / "reports/figures"

    # Muat dataset testing
    test_df = pd.read_csv(test_path)

    # Muat model
    if not model_path.exists():
        print(f"ERROR: File model {model_path} tidak ditemukan!")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Tentukan kolom fitur dan target
    feature_cols = ['lag_1', 'lag_2', 'rolling_mean_3', 'growth_rate']
    target_col = 'target'

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    weeks = test_df['week']

    # Lakukan prediksi
    y_pred = model.predict(X_test)

    # Hitung MAE, RMSE, dan MAPE
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Perhitungan MAPE (Mean Absolute Percentage Error)
    y_true_val = y_test.values
    mape = np.mean(np.abs((y_true_val - y_pred) / (y_true_val + 1e-5))) * 100

    print("\nHasil Evaluasi pada Data Testing (Mingguan):")
    print(f"  - MAE:  Rp {mae:,.2f}")
    print(f"  - RMSE: Rp {rmse:,.2f}")
    print(f"  - MAPE: {mape:.2f}%")

    # Simpan ke JSON
    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }

    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\n[OK] Metrik evaluasi disimpan di: {metrics_output_path}")

    # Plot Aktual vs Prediksi
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(weeks, y_test, marker='o', label='Aktual', color='blue', markersize=8)
    plt.plot(weeks, y_pred, marker='x', label='Prediksi XGBoost', color='red', linestyle='--', markersize=8)
    plt.title('Kurva Perbandingan Aktual vs Prediksi (Test Set Mingguan)', fontsize=12)
    plt.xlabel('Minggu (Tanggal Mulai)', fontsize=10)
    plt.ylabel('Nominal (Rupiah)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    plt.tight_layout()

    chart_path = figures_dir / "actual_vs_predicted.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[OK] Grafik visualisasi disimpan di: {chart_path}")


if __name__ == "__main__":
    evaluate_weekly()
