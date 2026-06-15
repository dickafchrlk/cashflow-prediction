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
    "# Proyek Eksperimentasi & Pelatihan Model XGBoost Bulanan\n",
    "\n",
    "**Judul Proyek**: Prediksi Arus Kas Koperasi Menggunakan Algoritma XGBoost untuk Mendukung Pengambilan Keputusan Keuangan  \n",
    "**Deskripsi**: Notebook ini digunakan khusus untuk proses eksperimentasi, rekayasa fitur (agregasi bulanan & lagging), pemisahan data temporal (Train-Val-Test), serta pelatihan dan evaluasi model XGBoost Regressor."
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
    "## 2. Feature Engineering (Skala Bulanan)\n",
    "Sesuai ketentuan, dataset bulanan wajib memiliki kolom berikut:\n",
    "* `month`\n",
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
    "df['month_period'] = df['TANGGAL'].dt.to_period('M')\n",
    "\n",
    "# Agregasi bulanan\n",
    "monthly_df = df.groupby('month_period').agg({\n",
    "    'CASH_IN': 'sum',\n",
    "    'CASH_OUT': 'sum'\n",
    "}).reset_index()\n",
    "\n",
    "monthly_df.rename(columns={\n",
    "    'month_period': 'month',\n",
    "    'CASH_IN': 'cash_in',\n",
    "    'CASH_OUT': 'cash_out'\n",
    "}, inplace=True)\n",
    "\n",
    "# Hitung net_cash\n",
    "monthly_df['net_cash'] = monthly_df['cash_in'] - monthly_df['cash_out']\n",
    "monthly_df = monthly_df.sort_values('month').reset_index(drop=True)\n",
    "\n",
    "# Lags\n",
    "monthly_df['lag_1'] = monthly_df['net_cash'].shift(1)\n",
    "monthly_df['lag_2'] = monthly_df['net_cash'].shift(2)\n",
    "\n",
    "# Rolling mean 3 periode dari lag\n",
    "monthly_df['rolling_mean_3'] = monthly_df['net_cash'].shift(1).rolling(window=3).mean()\n",
    "\n",
    "# Growth rate dari lag 2 ke lag 1\n",
    "monthly_df['growth_rate'] = (monthly_df['lag_1'] - monthly_df['lag_2']) / (monthly_df['lag_2'].abs() + 1e-5)\n",
    "\n",
    "# Target (NetCash t+1)\n",
    "monthly_df['target'] = monthly_df['net_cash'].shift(-1)\n",
    "\n",
    "# Ubah format month ke string\n",
    "monthly_df['month'] = monthly_df['month'].astype(str)\n",
    "\n",
    "# Drop NaN akibat lagging & target shifting\n",
    "features_df = monthly_df.dropna().reset_index(drop=True)\n",
    "print(f\"Dimensi dataset fitur bulanan: {features_df.shape}\")\n",
    "features_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Pembagian Data (Temporal Split)\n",
    "Rasio pembagian data:\n",
    "* **Train = 70%** (4 baris pertama)\n",
    "* **Validation = 15%** (baris ke-5)\n",
    "* **Test = 15%** (baris ke-6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_df = features_df.iloc[:4].reset_index(drop=True)\n",
    "val_df = features_df.iloc[4:5].reset_index(drop=True)\n",
    "test_df = features_df.iloc[5:].reset_index(drop=True)\n",
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
    "## 4. Pelatihan Model XGBoost Bulanan"
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
    "mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100\n",
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
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(test_df['month'], y_test, marker='o', label='Aktual', color='blue')\n",
    "plt.plot(test_df['month'], y_pred, marker='x', label='Prediksi XGBoost', color='red', linestyle='--')\n",
    "plt.title('Kurva Perbandingan Aktual vs Prediksi Bulanan')\n",
    "plt.xlabel('Bulan')\n",
    "plt.ylabel('Nominal (Rupiah)')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
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
    "print(\"Serialisasi selesai! model.pkl dan features.pkl berhasil disimpan di folder models.\")"
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
