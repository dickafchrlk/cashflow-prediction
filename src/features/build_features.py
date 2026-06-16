from pathlib import Path
import pandas as pd
import numpy as np

# Base directory setup
BASE_DIR = Path(__file__).resolve().parents[2]


def build_weekly_features(input_path, output_path):
    """ETL 2: Membaca clean_transactions.csv, melakukan agregasi mingguan, 
    dan merekayasa fitur-fitur time-series bulanan/mingguan secara aman."""
    print(f"Membaca data bersih dari: {input_path}...")
    df = pd.read_csv(input_path)
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])

    # Ekstrak Tahun-Minggu untuk agregasi mingguan
    df['week_period'] = df['TANGGAL'].dt.to_period('W')

    # Agregasi mingguan
    weekly_df = df.groupby('week_period').agg({
        'CASH_IN': 'sum',
        'CASH_OUT': 'sum'
    }).reset_index()

    weekly_df.rename(columns={
        'week_period': 'week',
        'CASH_IN': 'cash_in',
        'CASH_OUT': 'cash_out'
    }, inplace=True)

    # Hitung net_cash
    weekly_df['net_cash'] = weekly_df['cash_in'] - weekly_df['cash_out']

    # Urutkan secara kronologis
    weekly_df = weekly_df.sort_values('week').reset_index(drop=True)

    # 1. lag_1 (net_cash minggu lalu)
    weekly_df['lag_1'] = weekly_df['net_cash'].shift(1)

    # 2. lag_2 (net_cash 2 minggu lalu)
    weekly_df['lag_2'] = weekly_df['net_cash'].shift(2)

    # 3. rolling_mean_3 (rata-rata bergerak 3 minggu dari lag)
    # Untuk menghindari data leakage, rolling mean dihitung dari lag 1 (t-1, t-2, t-3)
    weekly_df['rolling_mean_3'] = weekly_df['net_cash'].shift(1).rolling(window=3).mean()

    # 4. growth_rate (persentase pertumbuhan dari lag 2 ke lag 1)
    weekly_df['growth_rate'] = (weekly_df['lag_1'] - weekly_df['lag_2']) / (weekly_df['lag_2'].abs() + 1e-5)

    # 5. target (arus kas bersih minggu berikutnya: Target(t) = NetCash(t+1))
    weekly_df['target'] = weekly_df['net_cash'].shift(-1)

    # Ubah format week ke string menggunakan start_time
    weekly_df['week'] = weekly_df['week'].apply(lambda r: r.start_time.strftime('%Y-%m-%d'))

    # Drop baris dengan nilai NaN (3 minggu pertama karena lag & rolling mean, dan 1 minggu terakhir karena target shift)
    final_df = weekly_df.dropna().reset_index(drop=True)

    # Pastikan folder output sudah ada
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    # Lakukan temporal split (70% Train, 15% Val, 15% Test) secara dinamis
    n = len(final_df)
    train_size = int(round(n * 0.70))
    val_size = int(round(n * 0.15))

    train_df = final_df.iloc[:train_size].reset_index(drop=True)
    val_df = final_df.iloc[train_size:train_size+val_size].reset_index(drop=True)
    test_df = final_df.iloc[train_size+val_size:].reset_index(drop=True)

    train_path = output_path.parent / "train.csv"
    val_path = output_path.parent / "val.csv"
    test_path = output_path.parent / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[OK] Sukses merekayasa fitur mingguan bulanan. Tersimpan di: {output_path} ({final_df.shape[0]} baris)")
    print(f"   - Train: {train_df.shape[0]} baris ({train_df['week'].min()} s/d {train_df['week'].max()})")
    print(f"   - Val: {val_df.shape[0]} baris ({val_df['week'].min()} s/d {val_df['week'].max()})")
    print(f"   - Test: {test_df.shape[0]} baris ({test_df['week'].min()} s/d {test_df['week'].max()})")
    return final_df


if __name__ == "__main__":
    input_file = BASE_DIR / "data/cleaned/clean_transactions.csv"
    output_file = BASE_DIR / "data/features/features.csv"
    build_weekly_features(input_file, output_file)
