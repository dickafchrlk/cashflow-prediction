# Prediksi Arus Kas Koperasi Menggunakan Algoritma XGBoost untuk Mendukung Pengambilan Keputusan Keuangan

Proyek ini bertujuan untuk membangun sistem prediksi arus kas koperasi berbasis Machine Learning tradisional (XGBoost Regressor) untuk memproyeksikan arus kas bersih (net cashflow) pada bulan berikutnya guna membantu pengambilan keputusan keuangan oleh pengurus koperasi.

---

## 📋 1. Latar Belakang
Pengelolaan arus kas (*cash flow*) yang sehat adalah pilar utama keberlangsungan operasional koperasi. Fluktuasi arus kas masuk (*cash inflow*) dan keluar (*cash outflow*) harian yang tidak menentu seringkali menyulitkan pengurus dalam merencanakan alokasi dana talangan, pembagian Sisa Hasil Usaha (SHU), maupun penyediaan likuiditas pinjaman anggota. Melalui proyek Machine Learning ini, riwayat transaksi dari jurnal kas dikonversi menjadi data runtun waktu bulanan untuk meramalkan kondisi arus kas bersih pada periode $t+1$.

---

## 📊 2. Dataset
Data diambil dari jurnal internal koperasi yang terbagi menjadi dua dokumen:
1. **JPNKAS (Jurnal Penerimaan Kas)**: Mencatat seluruh uang masuk koperasi (penjualan, setoran, bunga bank, dll.).
2. **JPNGKAS (Jurnal Pengeluaran Kas)**: Mencatat seluruh uang keluar koperasi (pembelian barang pokok, biaya operasional, gaji, ATK, dll.).

Rentang waktu transaksi kontinu utama yang dianalisis adalah dari **November 2024 s/d Oktober 2025**.

---

## 🧹 3. Data Cleaning (Pembersihan Data)
Tahap pembersihan data diimplementasikan secara otomatis dalam skrip `src/preprocessing/clean_data.py`:
* **Kolom kosong & Unnamed**: Kolom kosong (NaN) dan kolom teknis (`Unnamed`, `Column`) dibuang.
* **Baris Agregasi (TOTAL)**: Mendeteksi kata kunci `TOTAL` di jurnal kas penerimaan dan memangkas baris ringkasan tersebut agar tidak bocor sebagai pencatatan transaksi normal.
* **Konversi Tipe Data**: Nilai numerik dibersihkan dari format string (koma ribuan, simbol mata uang, tanda strip, teks placeholder seperti "QRIS BNI") dan di-cast menjadi `float64`.
* **Transformasi & Penggabungan**: Data penerimaan dan pengeluaran diagregasi per tanggal, lalu digabungkan secara temporal menggunakan outer-join untuk membentuk satu file transaksi terpadu.
* **Output Berkas Bersih**: Menghasilkan berkas tunggal **`data/cleaned/clean_transactions.csv`**.

---

## ⚙️ 4. Feature Engineering (Rekayasa Fitur)
Rekayasa fitur diimplementasikan di `src/feature_engineering/create_dataset.py`. Data transaksi harian dikompresi menjadi skala bulanan dengan melahirkan fitur-fitur wajib berikut:
1. `month`: Periode bulan (format `YYYY-MM`).
2. `cash_in`: Total kas masuk bulanan (penjumlahan dari `CASH_IN` harian).
3. `cash_out`: Total kas keluar bulanan (penjumlahan dari `CASH_OUT` harian).
4. `net_cash`: Selisih kas (`cash_in - cash_out`).
5. `lag_1`: Arus kas bersih bulan lalu ($t-1$).
6. `lag_2`: Arus kas bersih 2 bulan lalu ($t-2$).
7. `rolling_mean_3`: Rata-rata bergerak dari arus kas bersih 3 periode sebelumnya (menggunakan lag 1, 2, dan 3 untuk menghindari kebocoran data).
8. `growth_rate`: Kecepatan pertumbuhan kas dari $t-2$ ke $t-1$: $(lag\_1 - lag\_2) / lag\_2$.
9. `target`: Target prediksi arus kas bulan depan ($t+1$) $\rightarrow$ $\text{Target}(t) = \text{NetCash}(t+1)$.

---

## 🏋️ 5. Training Model
Proses pelatihan model diimplementasikan di `src/training/train_model.py` dan didokumentasikan di notebook eksperimental `notebooks/training.ipynb`:
* **Temporal Split**: Pembagian data dilakukan secara kronologis (tidak acak) dengan rasio:
  * **Train (70%)**: 4 baris pertama (Februari 2025 s/d Juli 2025).
  * **Validation (15%)**: 1 baris (Agustus 2025).
  * **Test (15%)**: 1 baris (September 2025).
* **Model**: Menggunakan **XGBRegressor** dengan hyperparameter terkontrol (`n_estimators=15`, `max_depth=2`, `learning_rate=0.05`) untuk mencegah overfitting pada sampel terbatas.

---

## 📈 6. Evaluasi Model
Model diuji pada data test set (September 2025) dengan metrik wajib berikut:
* **MAE (Mean Absolute Error)**: Rp 24,205,655.97
* **RMSE (Root Mean Squared Error)**: Rp 24,205,655.97
* **MAPE (Mean Absolute Percentage Error)**: 98.63%

*Catatan: Grafik visualisasi perbandingan Aktual vs Prediksi disimpan secara otomatis di berkas `reports/figures/actual_vs_predicted.png`.*

---

## 💾 7. Serialisasi Model
Setelah proses training selesai, objek model biner dan metadata fitur disimpan ke disk menggunakan format serialisasi Python standard `pickle`:
* **`models/model.pkl`**: Berkas biner model XGBoost hasil latih.
* **`models/features.pkl`**: Berkas list kolom fitur yang digunakan untuk inferensi.

---

## 🚀 8. Deployment (Aplikasi Streamlit)
Aplikasi Streamlit dibangun di `app/app.py` sebagai dashboard production:
* **Fitur Utama**:
  1. **Upload Dataset**: Pengguna dapat mengunggah berkas `clean_transactions.csv` terbaru.
  2. **Prediksi Otomatis**: Model membaca baris bulan terakhir untuk menghitung fitur lag/rolling, lalu memproyeksikan arus kas untuk bulan berikutnya ($\text{Target}(t) = \text{NetCash}(t+1)$).
  3. **Visualisasi Interaktif**: Grafik garis interaktif (menggunakan Plotly) yang menggabungkan tren arus kas historis dengan titik proyeksi bulan depan (garis merah putus-putus).
  4. **Tabel Fitur**: Tabel interaktif untuk meninjau detail rekayasa fitur secara transparan.

* **Cara Menjalankan Aplikasi Lokal**:
  # Install Virtual environment
  python -m venv venv
  ```powershell
  # Aktifkan virtual environment
  .\venv\Scripts\Activate.ps1
  
  # Jalankan Streamlit
  streamlit run app/app.py
  ```

---

## 📁 9. Struktur Proyek
```text
project/
├── app/
│   └── app.py                      # Berkas deployment Streamlit
├── data/
│   ├── raw/                        # Data jurnal mentah (JPNKAS.csv, JPNGKAS.csv)
│   ├── cleaned/                    # Data bersih terpadu (clean_transactions.csv)
│   └── features/                   # Dataset rekayasa fitur bulanan (features.csv)
├── models/
│   ├── model.pkl                   # Model XGBoost bulanan tersimpan (Pickle)
│   └── features.pkl                # Daftar nama fitur tersimpan (Pickle)
├── notebooks/
│   └── training.ipynb              # Notebook eksperimen, pelatihan & evaluasi (.ipynb)
├── src/
│   ├── preprocessing/
│   │   └── clean_data.py           # Script data cleaning
│   ├── feature_engineering/
│   │   └── create_dataset.py       # Script feature engineering
│   ├── training/
│   │   ├── train_test_split.py     # Script split data
│   │   └── train_model.py          # Script training model
│   └── evaluation/
│       └── evaluate_model.py       # Script kalkulasi metrik & plotting
├── reports/
│   ├── figures/                    # Output visualisasi chart
│   └── metrics.json                # JSON metrik evaluasi
├── requirements.txt                # Dependensi pustaka proyek
└── README.md                       # Dokumentasi proyek (berkas ini)
```
