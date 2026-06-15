from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]


def clean_csv(input_file, output_file):
    """Membersihkan data transaksi secara robust untuk kebutuhan machine learning."""
    print(f"Membaca {input_file}...")
    df = pd.read_csv(input_file)

    # 1. Merapikan nama kolom (hapus spasi berlebih)
    df.columns = df.columns.astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

    # 2. Hapus kolom Unnamed
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # 3. Identifikasi dan hapus baris TOTAL / footer Excel
    # Rule A: Potong data dari baris pertama yang memuat kata "TOTAL" di kolom URAIAN
    if 'URAIAN' in df.columns:
        total_mask = df['URAIAN'].astype(str).str.upper().str.strip() == 'TOTAL'
        if total_mask.any():
            first_total_idx = df[total_mask].index[0]
            print(f"-> Ditemukan baris 'TOTAL' pada indeks {first_total_idx}. Memotong baris setelahnya.")
            df = df.iloc[:first_total_idx]

    # Rule B: Buang baris di mana URAIAN dan BULAN sama-sama kosong (sering terjadi di footer JPNGKAS)
    if 'URAIAN' in df.columns and 'BULAN' in df.columns:
        footer_mask = df['URAIAN'].isna() & df['BULAN'].isna()
        if footer_mask.any():
            print(f"-> Menghapus {footer_mask.sum()} baris footer kosong (URAIAN & BULAN bernilai NaN).")
            df = df[~footer_mask]

    # 4. Hapus baris di mana kolom TANGGAL kosong (runtun waktu membutuhkan tanggal yang valid)
    if 'TANGGAL' in df.columns:
        null_dates = df['TANGGAL'].isna()
        if null_dates.any():
            print(f"-> Menghapus {null_dates.sum()} baris dengan TANGGAL bernilai NaN.")
            df = df[~null_dates]

    # 5. Membersihkan nilai numerik dari format string (seperti koma, spasi ganda, teks seperti "QRIS BNI", atau "-")
    non_numeric_cols = ['TANGGAL', 'URAIAN', 'BULAN']
    numeric_cols = [c for c in df.columns if c not in non_numeric_cols]

    for col in numeric_cols:
        # Bersihkan karakter non-numerik (kecuali angka, titik desimal, dan tanda minus)
        cleaned_series = df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
        # Konversi ke numerik, ganti nilai tidak valid menjadi NaN, lalu isi dengan 0.0
        df[col] = pd.to_numeric(cleaned_series, errors='coerce').fillna(0.0)

    # 6. Hapus duplikasi baris
    df = df.drop_duplicates()

    # 7. Hapus baris yang seluruh kolomnya kosong
    df = df.dropna(how="all")

    # Reset indeks setelah pemotongan/pembersihan baris
    df.reset_index(drop=True, inplace=True)

    # Simpan hasil pembersihan ke output file
    df.to_csv(output_file, index=False)
    print(f"[OK] Selesai membersihkan data. Tersimpan di: {output_file} (Ukuran: {df.shape[0]} baris x {df.shape[1]} kolom)\n")


if __name__ == "__main__":
    clean_csv(
        BASE_DIR / "data/raw/JPNKAS.csv",
        BASE_DIR / "data/cleaned/JPNKAS_clean.csv",
    )

    clean_csv(
        BASE_DIR / "data/raw/JPNGKAS.csv",
        BASE_DIR / "data/cleaned/JPNGKAS_clean.csv",
    )