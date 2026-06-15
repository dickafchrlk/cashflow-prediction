from pathlib import Path
import pandas as pd
import numpy as np

# Base directory setup
BASE_DIR = Path(__file__).resolve().parents[2]


def build_monthly_features():
    print("Memulai feature engineering skala bulanan sesuai ketentuan tugas...")
    
    clean_transactions_path = BASE_DIR / "data/cleaned/clean_transactions.csv"
    output_path = BASE_DIR / "data/features/features.csv" # Simpan sebagai features.csv

    # Muat data transaksi bersih
    df = pd.read_csv(clean_transactions_path)
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])

    # Ekstrak Tahun-Bulan untuk agregasi bulanan
    df['month_period'] = df['TANGGAL'].dt.to_period('M')

    # Agregasi bulanan
    monthly_df = df.groupby('month_period').agg({
        'CASH_IN': 'sum',
        'CASH_OUT': 'sum'
    }).reset_index()

    monthly_df.rename(columns={
        'month_period': 'month',
        'CASH_IN': 'cash_in',
        'CASH_OUT': 'cash_out'
    }, inplace=True)

    # Hitung net_cash
    monthly_df['net_cash'] = monthly_df['cash_in'] - monthly_df['cash_out']

    # Urutkan secara kronologis
    monthly_df = monthly_df.sort_values('month').reset_index(drop=True)

    # 1. lag_1 (net_cash bulan lalu)
    monthly_df['lag_1'] = monthly_df['net_cash'].shift(1)

    # 2. lag_2 (net_cash 2 bulan lalu)
    monthly_df['lag_2'] = monthly_df['net_cash'].shift(2)

    # 3. rolling_mean_3 (rata-rata bergerak 3 bulan dari lag)
    # Untuk menghindari data leakage, rolling mean dihitung dari lag 1 (yaitu dari t-1, t-2, t-3)
    monthly_df['rolling_mean_3'] = monthly_df['net_cash'].shift(1).rolling(window=3).mean()

    # 4. growth_rate (persentase pertumbuhan dari lag 2 ke lag 1)
    monthly_df['growth_rate'] = (monthly_df['lag_1'] - monthly_df['lag_2']) / (monthly_df['lag_2'].abs() + 1e-5)

    # 5. target (arus kas bersih bulan berikutnya: Target(t) = NetCash(t+1))
    monthly_df['target'] = monthly_df['net_cash'].shift(-1)

    # Ubah format month ke string (YYYY-MM)
    monthly_df['month'] = monthly_df['month'].astype(str)

    # Drop baris dengan nilai NaN (3 bulan pertama karena lag & rolling mean, dan 1 bulan terakhir karena target shift)
    final_df = monthly_df.dropna().reset_index(drop=True)

    # Simpan dataset fitur final
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"\n[OK] Sukses membuat dataset fitur bulanan di: {output_path}")
    print(f"     Dimensi: {final_df.shape[0]} baris x {final_df.shape[1]} kolom")
    print(final_df.to_string())
    print("\nKolom yang terbentuk:")
    print(list(final_df.columns))


if __name__ == "__main__":
    build_monthly_features()
