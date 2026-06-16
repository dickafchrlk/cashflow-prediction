import pandas as pd
import numpy as np
import pytest
from src.preprocessing.clean_data import clean_cash_journals


def test_clean_cash_journals_removes_unnamed_and_totals():
    # Mock JPNKAS (Penerimaan)
    df_kas = pd.DataFrame({
        'TANGGAL': ['2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05'],
        'URAIAN': ['penerimaan', 'penerimaan', 'TOTAL', ''],
        'DBT_KAS': ['5,236,000', '3,733,000', '8,969,000', ''],
        'Unnamed: 0': [1, 2, 3, 4]
    })

    # Mock JPNGKAS (Pengeluaran)
    df_gkas = pd.DataFrame({
        'TANGGAL': ['2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05'],
        'URAIAN': ['pengeluaran', 'pengeluaran', '', ''],
        'BULAN': ['JANUARI', 'JANUARI', '', ''],
        'KREDIT_KAS': ['3,426,900', '5,458,000', '', '']
    })

    merged_df = clean_cash_journals(df_kas, df_gkas)

    # 1. Pastikan kolom Unnamed dihapus
    assert not any('Unnamed' in col for col in merged_df.columns)

    # 2. Pastikan baris rekapitulasi TOTAL dihapus (hanya ada tanggal 2 dan 3 yang valid)
    assert len(merged_df) == 2
    assert '2025-01-02' in merged_df['TANGGAL'].values
    assert '2025-01-03' in merged_df['TANGGAL'].values
    assert '2025-01-04' not in merged_df['TANGGAL'].values # baris TOTAL

    # 3. Pastikan kolom numerik terkonversi ke float dengan benar
    assert merged_df.loc[merged_df['TANGGAL'] == '2025-01-02', 'CASH_IN'].iloc[0] == 5236000.0
    assert merged_df.loc[merged_df['TANGGAL'] == '2025-01-02', 'CASH_OUT'].iloc[0] == 3426900.0
    assert merged_df.loc[merged_df['TANGGAL'] == '2025-01-02', 'NET_CASH'].iloc[0] == 1809100.0
