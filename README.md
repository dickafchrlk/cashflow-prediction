# Prediksi Arus Kas Koperasi Menggunakan Algoritma XGBoost untuk Mendukung Pengambilan Keputusan Keuangan

Aplikasi *Machine Learning* berbasis web ini berfungsi untuk memprediksi arus kas bersih (*net cashflow*) koperasi pada periode berikutnya ($t+1$) dengan menganalisis riwayat transaksi keuangan mingguan menggunakan model **XGBoost Regressor**. Sistem ini dirancang untuk membantu pengurus koperasi dalam mengambil keputusan keuangan strategis seperti pengelolaan dana talangan, penyediaan likuiditas pinjaman anggota, dan perencanaan anggaran mingguan.

---

## 🛠️ 1. Prasyarat (Prerequisites)

Sebelum menjalankan proyek ini di komputer lokal Anda, pastikan sistem Anda telah memenuhi prasyarat berikut:

* **Sistem Operasi**: Windows (disarankan), Linux, atau macOS.
* **Python**: Versi **Python 3.10** s/d **Python 3.12** (proyek ini diuji pada Python 3.12).
* **Pip**: Versi terbaru (manajer paket bawaan Python).
* **Git**: Digunakan untuk cloning repositori.
* **Perangkat Lunak Pendukung**: Jupyter Notebook (opsional, untuk membuka berkas eksperimen `.ipynb`).

---

## ⚙️ 2. Cara Instalasi (Installation)

Ikuti langkah-langkah teknis berikut untuk memasang proyek di lingkungan lokal Anda:

### Langkah 1: Kloning Repositori
Buka terminal (PowerShell, Command Prompt, atau Git Bash) dan jalankan perintah:
```bash
git clone https://github.com/dickafchrlk/cashflow-prediction.git
cd cashflow-prediction
```

### Langkah 2: Buat Lingkungan Virtual (Virtual Environment)
Untuk mencegah bentrok antar pustaka, buat lingkungan virtual terisolasi bernama `venv`:
```powershell
# Di Windows (PowerShell)
python -m venv venv

# Di macOS / Linux
python3 -m venv venv
```

### Langkah 3: Aktifkan Lingkungan Virtual
Aktifkan `venv` sebelum menginstal pustaka apa pun:
```powershell
# Di Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Di Windows (Command Prompt)
.\venv\Scripts\activate.bat

# Di macOS / Linux
source venv/bin/activate
```

### Langkah 4: Pasang Semua Dependensi Pustaka
Pasang pustaka yang tercantum pada berkas `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📈 3. Aliran & Pipeline Data

Proyek ini mengikuti arsitektur alur data terstruktur:
```text
RAW DATA (JPNKAS & JPNGKAS)
    ↓ (clean_data.py)
CLEANED DATA (clean_transactions.csv)
    ↓ (build_features.py)
FEATURE DATASET & SPLITS (features.csv, train.csv, val.csv, test.csv)
    ↓ (train.py)
SERIALIZED BINARY (model.pkl, features.pkl)
    ↓ (evaluate.py)
EVALUATION METRICS & FIGURES (metrics.json, actual_vs_predicted.png)
    ↓
STREAMLIT DEPLOYMENT (app.py)
```

---

## 🧹 4. Data Cleaning (Pembersihan Data)
Tahap pembersihan data dijalankan melalui perintah `python src/preprocessing/clean_data.py`:
* **Pembersihan Kolom**: Kolom-kolom kosong (NaN) dan kolom teknis spreadsheet (`Unnamed`, `Column`) dihapus.
* **Pembersihan Baris TOTAL & String Kosong**: Baris berisi penjumlahan rekapitulasi `TOTAL` dideteksi secara dinamis dan dipotong. Seluruh sel string kosong atau spasi dibersihkan dan diubah menjadi `NaN`.
* **Konversi & Casting**: Data numerik dibersihkan dari format string (koma ribuan, simbol mata uang, teks placeholder) dan di-cast menjadi `float64`.
* **Output**: Jurnal kas penerimaan (`JPNKAS`) dan pengeluaran (`JPNGKAS`) digabungkan secara harian, lalu disimpan ke dalam berkas tunggal **`data/cleaned/clean_transactions.csv`**.

---

## ⚙️ 5. Feature Engineering & Split (Rekayasa Fitur & Pembagian Data)
Pembuatan fitur mingguan dan pembagian data dijalankan melalui perintah `python src/features/build_features.py`. Dataset akhir memiliki fitur-fitur wajib berikut:
* `week`: Periode minggu (format tanggal awal minggu `YYYY-MM-DD`).
* `cash_in`: Total kas masuk mingguan.
* `cash_out`: Total kas keluar mingguan.
* `net_cash`: Selisih kas mingguan (`cash_in - cash_out`).
* `lag_1`: Arus kas bersih minggu lalu ($t-1$).
* `lag_2`: Arus kas bersih 2 minggu lalu ($t-2$).
* `rolling_mean_3`: Rata-rata bergerak dari arus kas bersih 3 periode sebelumnya (lag 1, 2, dan 3).
* `growth_rate`: Tingkat pertumbuhan kas dari $t-2$ ke $t-1$: $(lag\_1 - lag\_2) / (abs(lag\_2) + 1e-5)$.
* `target`: Arus kas bersih periode berikutnya $\rightarrow$ $\text{Target}(t) = \text{NetCash}(t+1)$.

*Catatan: Baris data awal yang memiliki nilai kosong akibat lagging (3 minggu pertama) dan baris akhir yang kosong akibat pergeseran target otomatis dibuang.*

**Pemisahan Temporal Otomatis**:
Setelah rekayasa fitur selesai, data secara otomatis dipisah secara kronologis dengan rasio **Train (70%)**, **Validation (15%)**, dan **Test (15%)**:
* **Train (70%)**: 27 baris pertama (2025-01-20 s/d 2025-07-28) disimpan di `data/features/train.csv`.
* **Validation (15%)**: 6 baris (2025-08-04 s/d 2025-09-08) disimpan di `data/features/val.csv`.
* **Test (15%)**: 6 baris (2025-09-15 s/d 2025-10-20) disimpan di `data/features/test.csv`.

---

## 🏋️ 6. Pelatihan & Evaluasi Model
* **Pelatihan Model**: Dijalankan melalui `python src/models/train.py`. Melakukan komparasi model Baseline, Linear Regression, Random Forest, dan XGBoost menggunakan `TimeSeriesSplit` (n_splits=3) dan `GridSearchCV`. Hyperparameter model XGBoost terbaik disimpan secara serial ke `models/model.pkl` dan list fitur ke `models/features.pkl`.
* **Evaluasi Akhir**: Dijalankan melalui `python src/evaluation/evaluate.py`. Melakukan pengujian pada Test Set dan menyimpan laporan metrik ke `reports/metrics.json` serta visualisasi grafik ke folder `reports/figures/`.
* **Metrik Evaluasi Akhir (Test Set)**:
  * **MAE**: Rp 4,207,752.02
  * **RMSE**: Rp 8,723,658.63
  * **MAPE**: 133.99%
  * **R²**: 0.0601
* **Penyimpanan (Serialisasi)**: Model biner disimpan ke **`models/model.pkl`** dan kolom fitur disimpan ke **`models/features.pkl`**.

---

## 🚀 7. Cara Menjalankan Aplikasi Web Streamlit

Setelah semua prasyarat dan instalasi di atas selesai dilakukan, jalankan perintah berikut di terminal Anda untuk membuka dashboard interaktif:

```powershell
# Pastikan virtual environment teraktivasi
& venv/Scripts/python.exe -m streamlit run app/app.py
```

Setelah aplikasi berjalan, buka peramban (*web browser*) Anda ke alamat lokal:
`http://localhost:8501`

**Fitur Dashboard**:
1. **Pemuatan Otomatis**: Membaca dataset mingguan default secara otomatis.
2. **Unggah Berkas Baru**: Anda dapat mengunggah berkas `clean_transactions.csv` terbaru dari komputer lokal Anda.
3. **Prediksi Arus Kas Bersih**: Model akan memproyeksikan arus kas bersih minggu depan secara otomatis.
4. **Grafik Interaktif**: Menggambarkan grafik historis dan titik hasil proyeksi minggu depan menggunakan Plotly.
