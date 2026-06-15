from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]


def clean_and_merge():
    print("Memulai pembersihan data...")
    raw_jpnkas_path = BASE_DIR / "data/raw/JPNKAS.csv"
    raw_jpngkas_path = BASE_DIR / "data/raw/JPNGKAS.csv"
    output_path = BASE_DIR / "data/cleaned/clean_transactions.csv"

    # Muat dataset mentah
    df_kas = pd.read_csv(raw_jpnkas_path)
    df_gkas = pd.read_csv(raw_jpngkas_path)

    # 1. Rapikan nama kolom (hapus spasi berlebih)
    df_kas.columns = df_kas.columns.astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    df_gkas.columns = df_gkas.columns.astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

    # 2. Hapus kolom Unnamed
    df_kas = df_kas.loc[:, ~df_kas.columns.str.contains("^Unnamed")]
    df_gkas = df_gkas.loc[:, ~df_gkas.columns.str.contains("^Unnamed")]

    # 3. Hapus baris agregasi (TOTAL) di dasar file
    if 'URAIAN' in df_kas.columns:
        total_mask = df_kas['URAIAN'].astype(str).str.upper().str.strip() == 'TOTAL'
        if total_mask.any():
            first_total_idx = df_kas[total_mask].index[0]
            df_kas = df_kas.iloc[:first_total_idx]

    if 'URAIAN' in df_gkas.columns and 'BULAN' in df_gkas.columns:
        footer_mask = df_gkas['URAIAN'].isna() & df_gkas['BULAN'].isna()
        if footer_mask.any():
            df_gkas = df_gkas[~footer_mask]

    # 4. Hapus baris dengan tanggal kosong
    if 'TANGGAL' in df_kas.columns:
        df_kas = df_kas[df_kas['TANGGAL'].notna()]
    if 'TANGGAL' in df_gkas.columns:
        df_gkas = df_gkas[df_gkas['TANGGAL'].notna()]

    # 5. Membersihkan data numerik (koma, tanda strip, format string)
    # JPNKAS: kolom penerimaan utama DBT_KAS
    cleaned_dbt = df_kas['DBT_KAS'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
    df_kas['DBT_KAS'] = pd.to_numeric(cleaned_dbt, errors='coerce').fillna(0.0)

    # JPNGKAS: kolom pengeluaran utama KREDIT_KAS
    cleaned_kredit = df_gkas['KREDIT_KAS'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
    df_gkas['KREDIT_KAS'] = pd.to_numeric(cleaned_kredit, errors='coerce').fillna(0.0)

    # 6. Agregasi data berdasarkan TANGGAL dan BULAN
    df_kas['TANGGAL'] = pd.to_datetime(df_kas['TANGGAL'])
    df_gkas['TANGGAL'] = pd.to_datetime(df_gkas['TANGGAL'])

    daily_inflow = df_kas.groupby(['TANGGAL', 'BULAN'])['DBT_KAS'].sum().reset_index()
    daily_inflow.rename(columns={'DBT_KAS': 'CASH_IN'}, inplace=True)

    # Catatan: JPNGKAS tidak selalu memiliki nama kolom BULAN yang lengkap di setiap baris, 
    # kita ambil BULAN dari JPNKAS jika memungkinkan, atau isi otomatis
    daily_outflow = df_gkas.groupby('TANGGAL')['KREDIT_KAS'].sum().reset_index()
    daily_outflow.rename(columns={'KREDIT_KAS': 'CASH_OUT'}, inplace=True)

    # 7. Merge menjadi satu file transaksi bersih
    merged_df = pd.merge(daily_inflow, daily_outflow, on='TANGGAL', how='outer')
    
    # Isi missing values akibat outer join dengan 0.0
    merged_df['CASH_IN'] = merged_df['CASH_IN'].fillna(0.0)
    merged_df['CASH_OUT'] = merged_df['CASH_OUT'].fillna(0.0)
    merged_df['NET_CASH'] = merged_df['CASH_IN'] - merged_df['CASH_OUT']

    # Filter rentang waktu yang valid dari 2025 onwards (sesuai data mayoritas)
    merged_df = merged_df[merged_df['TANGGAL'] >= '2025-01-01'].reset_index(drop=True)

    # Isi kolom BULAN yang kosong berdasarkan tanggal
    # (Pemetaan nama bulan Indonesia)
    months_id = {
        1: "JANUARI", 2: "FEBRUARI", 3: "MARET", 4: "APRIL", 5: "MEI", 6: "JUNI",
        7: "JULI", 8: "AGUSTUS", 9: "SEPTEMBER", 10: "OKTOBER", 11: "NOVEMBER", 12: "DESEMBER"
    }
    merged_df['BULAN'] = merged_df['TANGGAL'].dt.month.map(months_id)

    # Ubah format TANGGAL kembali ke string YYYY-MM-DD
    merged_df['TANGGAL'] = merged_df['TANGGAL'].dt.strftime('%Y-%m-%d')

    # Urutkan kronologis
    merged_df = merged_df.sort_values('TANGGAL').reset_index(drop=True)

    # Simpan hasil akhir
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    print(f"[OK] Sukses menyimpan data bersih tunggal di: {output_path}")
    print(f"     Dimensi: {merged_df.shape[0]} baris x {merged_df.shape[1]} kolom")
    print(merged_df.head(3))


if __name__ == "__main__":
    clean_and_merge()