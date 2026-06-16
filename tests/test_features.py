import pandas as pd
import numpy as np
import pytest
from src.features.build_features import build_weekly_features


def test_build_weekly_features_calculates_correct_columns(tmp_path):
    # Setup mock clean_transactions.csv
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    mock_clean_df = pd.DataFrame({
        'TANGGAL': dates,
        'CASH_IN': np.random.uniform(5000000, 10000000, size=100),
        'CASH_OUT': np.random.uniform(3000000, 7000000, size=100),
        'NET_CASH': [0.0] * 100,
        'BULAN': ['JANUARI'] * 100
    })
    mock_clean_df['NET_CASH'] = mock_clean_df['CASH_IN'] - mock_clean_df['CASH_OUT']

    input_file = tmp_path / "clean_transactions.csv"
    output_file = tmp_path / "features.csv"
    
    mock_clean_df.to_csv(input_file, index=False)

    # Run build_weekly_features
    final_df = build_weekly_features(input_file, output_file)

    # 1. Pastikan kolom fitur wajib terbentuk
    expected_cols = ['week', 'cash_in', 'cash_out', 'net_cash', 'lag_1', 'lag_2', 'rolling_mean_3', 'growth_rate', 'target']
    for col in expected_cols:
        assert col in final_df.columns

    # 2. Pastikan baris target(t) adalah net_cash(t+1)
    # Target baris ke-0 harus bernilai net_cash baris ke-1
    assert final_df['target'].iloc[0] == final_df['net_cash'].iloc[1]

    # 3. Pastikan rolling mean dihitung dengan benar dari lag 1
    # rolling_mean_3 untuk indeks ke-3 adalah mean dari net_cash indeks 0, 1, 2
    expected_roll = final_df['net_cash'].iloc[0:3].mean()
    assert np.isclose(final_df['rolling_mean_3'].iloc[3], expected_roll)
