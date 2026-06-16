import json
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Advan\Documents\Kuliah\Semester 4\Machine Learning\deploymen\cashflow-Prediction")
notebook_path = BASE_DIR / "notebooks/training.ipynb"

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Proyek Eksperimentasi & Pelatihan Model XGBoost Mingguan\n",
    "\n",
    "**Judul Proyek**: Prediksi Arus Kas Koperasi Menggunakan Algoritma XGBoost untuk Mendukung Pengambilan Keputusan Keuangan  \n",
    "**Deskripsi**: Notebook ini digunakan khusus untuk proses eksperimentasi, rekayasa fitur (agregasi mingguan & lagging), pemisahan data temporal (Train-Val-Test), serta pelatihan dan evaluasi model XGBoost Regressor."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pickle\n",
    "import os\n",
    "from xgboost import XGBRegressor\n",
    "from sklearn.metrics import root_mean_squared_error, mean_absolute_error\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Dataset Hasil Pembersihan (`clean_transactions.csv`)\n",
    "Tahapan: `RAW` -> `cleanData` (yaitu `clean_transactions.csv`)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "clean_data_path = '../data/cleaned/clean_transactions.csv'\n",
    "df = pd.read_csv(clean_data_path)\n",
    "print(f\"Dimensi data transaksi bersih: {df.shape}\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Feature Engineering (Skala Mingguan)\n",
    "Sesuai ketentuan, dataset mingguan memiliki kolom fitur berikut:\n",
    "* `week`\n",
    "* `cash_in`\n",
    "* `cash_out`\n",
    "* `net_cash`\n",
    "* `lag_1`\n",
    "* `lag_2`\n",
    "* `rolling_mean_3`\n",
    "* `growth_rate`\n",
    "* `target` (dimana $\\text{Target}(t) = \\text{NetCash}(t+1)$)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])\n",
    "df['week_period'] = df['TANGGAL'].dt.to_period('W')\n",
    "\n",
    "# Agregasi mingguan\n",
    "weekly_df = df.groupby('week_period').agg({\n",
    "    'CASH_IN': 'sum',\n",
    "    'CASH_OUT': 'sum'\n",
    "}).reset_index()\n",
    "\n",
    "weekly_df.rename(columns={\n",
    "    'week_period': 'week',\n",
    "    'CASH_IN': 'cash_in',\n",
    "    'CASH_OUT': 'cash_out'\n",
    "}, inplace=True)\n",
    "\n",
    "# Hitung net_cash\n",
    "weekly_df['net_cash'] = weekly_df['cash_in'] - weekly_df['cash_out']\n",
    "weekly_df = weekly_df.sort_values('week').reset_index(drop=True)\n",
    "\n",
    "# Lags\n",
    "weekly_df['lag_1'] = weekly_df['net_cash'].shift(1)\n",
    "weekly_df['lag_2'] = weekly_df['net_cash'].shift(2)\n",
    "\n",
    "# Rolling mean 3 periode dari lag (t-1, t-2, t-3)\n",
    "weekly_df['rolling_mean_3'] = weekly_df['net_cash'].shift(1).rolling(window=3).mean()\n",
    "\n",
    "# Growth rate dari lag 2 ke lag 1\n",
    "weekly_df['growth_rate'] = (weekly_df['lag_1'] - weekly_df['lag_2']) / (weekly_df['lag_2'].abs() + 1e-5)\n",
    "\n",
    "# Target (NetCash t+1)\n",
    "weekly_df['target'] = weekly_df['net_cash'].shift(-1)\n",
    "\n",
    "# Ubah format week ke string menggunakan start_time\n",
    "weekly_df['week'] = weekly_df['week'].apply(lambda r: r.start_time.strftime('%Y-%m-%d'))\n",
    "\n",
    "# Drop NaN akibat lagging & target shifting\n",
    "features_df = weekly_df.dropna().reset_index(drop=True)\n",
    "print(f\"Dimensi dataset fitur mingguan: {features_df.shape}\")\n",
    "features_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Pembagian Data (Temporal Split)\n",
    "Rasio pembagian data:\n",
    "* **Train = 70%** (27 baris pertama)\n",
    "* **Validation = 15%** (6 baris berikutnya)\n",
    "* **Test = 15%** (6 baris terakhir)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_df = features_df.iloc[:27].reset_index(drop=True)\n",
    "val_df = features_df.iloc[27:33].reset_index(drop=True)\n",
    "test_df = features_df.iloc[33:].reset_index(drop=True)\n",
    "\n",
    "feature_cols = ['lag_1', 'lag_2', 'rolling_mean_3', 'growth_rate']\n",
    "target_col = 'target'\n",
    "\n",
    "X_train, y_train = train_df[feature_cols], train_df[target_col]\n",
    "X_val, y_val = val_df[feature_cols], val_df[target_col]\n",
    "X_test, y_test = test_df[feature_cols], test_df[target_col]\n",
    "\n",
    "print(f\"Ukuran X_train: {X_train.shape}\")\n",
    "print(f\"Ukuran X_val:   {X_val.shape}\")\n",
    "print(f\"Ukuran X_test:  {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Pelatihan Model XGBoost Mingguan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = XGBRegressor(\n",
    "    n_estimators=15,\n",
    "    max_depth=2,\n",
    "    learning_rate=0.05,\n",
    "    random_state=42,\n",
    "    objective='reg:squarederror'\n",
    ")\n",
    "\n",
    "model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Evaluasi Model pada Data Pengujian (Test Set)\n",
    "Metrik evaluasi wajib:\n",
    "* MAE\n",
    "* RMSE\n",
    "* MAPE (Mean Absolute Percentage Error)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(X_test)\n",
    "\n",
    "rmse = root_mean_squared_error(y_test, y_pred)\n",
    "mae = mean_absolute_error(y_test, y_pred)\n",
    "mape = np.mean(np.abs((y_test.values - y_pred) / (y_test.values + 1e-5))) * 100\n",
    "\n",
    "print(\"=== Hasil Evaluasi pada Test Set ===\")\n",
    "print(f\"MAE:  Rp {mae:,.2f}\")\n",
    "print(f\"RMSE: Rp {rmse:,.2f}\")\n",
    "print(f\"MAPE: {mape:.2f}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Visualisasi Aktual vs Prediksi"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 5))\n",
    "plt.plot(test_df['week'], y_test, marker='o', label='Aktual', color='blue')\n",
    "plt.plot(test_df['week'], y_pred, marker='x', label='Prediksi XGBoost', color='red', linestyle='--')\n",
    "plt.title('Kurva Perbandingan Aktual vs Prediksi Mingguan')\n",
    "plt.xlabel('Minggu (Tanggal Mulai)')\n",
    "plt.ylabel('Nominal (Rupiah)')\n",
    "plt.xticks(rotation=45)\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Serialisasi Model & Daftar Fitur (.pkl)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "os.makedirs('../models', exist_ok=True)\n",
    "with open('../models/model.pkl', 'wb') as f:\n",
    "    pickle.dump(model, f)\n",
    "\n",
    "with open('../models/features.pkl', 'wb') as f:\n",
    "    pickle.dump(feature_cols, f)\n",
    "print(\"Serialisasi selesai! model.pkl and features.pkl berhasil disimpan di folder models.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open(notebook_path, 'w') as f:
    json.dump(notebook_content, f, indent=1)

print(f"Jupyter Notebook successfully written to {notebook_path}")
