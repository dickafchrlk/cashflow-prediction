import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from pathlib import Path

# Setup page config
st.set_page_config(
    page_title="Prediksi Arus Kas Koperasi (Bulanan)",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
    /* Gradient Header */
    .header-container {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.8;
    }
    /* Metric Cards */
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #2c5364;
        padding: 1.2rem;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c5364;
        margin-top: 0.3rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Path definition
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models/model.pkl"
FEATURES_PKL_PATH = BASE_DIR / "models/features.pkl"
DEFAULT_DATA_PATH = BASE_DIR / "data/features/features.csv"


# Helper: Load serialized model
@st.cache_resource
def load_serialized_model():
    if MODEL_PATH.exists() and FEATURES_PKL_PATH.exists():
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(FEATURES_PKL_PATH, 'rb') as f:
            features = pickle.load(f)
        return model, features
    return None, None


# Helper: Load default features
def load_default_features():
    if DEFAULT_DATA_PATH.exists():
        df = pd.read_csv(DEFAULT_DATA_PATH)
        return df
    return None


# Helper: Dynamic feature engineering from clean transactions
def process_monthly_features(clean_df):
    clean_df['TANGGAL'] = pd.to_datetime(clean_df['TANGGAL'])
    clean_df['month_period'] = clean_df['TANGGAL'].dt.to_period('M')
    
    # Aggregate monthly
    monthly_df = clean_df.groupby('month_period').agg({
        'CASH_IN': 'sum',
        'CASH_OUT': 'sum'
    }).reset_index()
    
    monthly_df.rename(columns={
        'month_period': 'month',
        'CASH_IN': 'cash_in',
        'CASH_OUT': 'cash_out'
    }, inplace=True)
    
    monthly_df['net_cash'] = monthly_df['cash_in'] - monthly_df['cash_out']
    monthly_df = monthly_df.sort_values('month').reset_index(drop=True)
    
    # Lag and rolling features
    monthly_df['lag_1'] = monthly_df['net_cash'].shift(1)
    monthly_df['lag_2'] = monthly_df['net_cash'].shift(2)
    monthly_df['rolling_mean_3'] = monthly_df['net_cash'].shift(1).rolling(window=3).mean()
    monthly_df['growth_rate'] = (monthly_df['lag_1'] - monthly_df['lag_2']) / (monthly_df['lag_2'].abs() + 1e-5)
    
    monthly_df['month'] = monthly_df['month'].astype(str)
    return monthly_df


# Header Banner
st.markdown("""
<div class="header-container">
    <div class="header-title">💰 Dashboard Prediksi Arus Kas Bulanan Koperasi</div>
    <div class="header-subtitle">Aplikasi Production untuk Peramalan Arus Kas Bersih Bulan Depan menggunakan XGBoost Regressor.</div>
</div>
""", unsafe_allow_html=True)

# Load model and feature list
model, feature_cols = load_serialized_model()

if model is None:
    st.error("Model biner (`models/model.pkl`) atau daftar fitur (`models/features.pkl`) tidak ditemukan! Silakan jalankan script training terlebih dahulu.")
    st.stop()

# Sidebar: Config & Upload
st.sidebar.image("https://img.icons8.com/color/96/money-circulation.png", width=90)
st.sidebar.title("Konfigurasi & Input")

upload_mode = st.sidebar.checkbox("Unggah Dataset (CSV)", value=False, help="Centang untuk mengunggah berkas clean_transactions.csv terbaru.")

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Spesifikasi Model", expanded=True):
    st.markdown("""
    * **Model**: XGBoost Regressor
    * **Tingkat Waktu**: Bulanan (Monthly)
    * **Metrik Pelatihan**:
      * Train MAE: Rp 956,976.73
      * Test MAE: Rp 24,205,655.97
      * Test MAPE: 98.63%
    """)

# Data loading
features_df = None

if upload_mode:
    st.sidebar.subheader("Unggah clean_transactions.csv")
    uploaded_file = st.sidebar.file_uploader("Pilih file CSV bersih", type="csv")
    
    if uploaded_file:
        try:
            raw_clean_df = pd.read_csv(uploaded_file)
            features_df = process_monthly_features(raw_clean_df)
            st.sidebar.success("✅ Dataset bulanan berhasil direkayasa!")
        except Exception as e:
            st.sidebar.error(f"Error memproses file: {e}")
    else:
        st.sidebar.warning("Silakan unggah file clean_transactions.csv untuk memulai.")
else:
    features_df = load_default_features()
    if features_df is None:
        st.warning("Data default `data/features/features.csv` tidak ditemukan. Silakan gunakan opsi Unggah Dataset.")

if features_df is not None:
    # 1. Prediction for the next month
    # We build the features of the last month to predict the next month's target (Target(t) = NetCash(t+1))
    # Note: If the last row in the uploaded dataset has NaN in target, it means we don't know the actual net_cash of next month yet, 
    # but we can build features for the last month to predict it!
    
    # Check if there is a row with NaNs that we need to predict
    # Let's take the very last month that has complete lag/rolling features to predict the next month
    latest_valid_idx = features_df.dropna(subset=['lag_1', 'lag_2', 'rolling_mean_3']).index[-1]
    latest_row = features_df.loc[latest_valid_idx]
    
    # Calculate next month
    latest_month_period = pd.Period(latest_row['month'], freq='M')
    next_month_period = latest_month_period + 1
    next_month_str = str(next_month_period)
    
    # Build feature vector for prediction
    pred_features = pd.DataFrame([{
        'lag_1': latest_row['net_cash'],
        'lag_2': latest_row['lag_1'],
        'rolling_mean_3': (latest_row['net_cash'] + latest_row['lag_1'] + latest_row['lag_2']) / 3.0,
        'growth_rate': (latest_row['net_cash'] - latest_row['lag_1']) / (abs(latest_row['lag_1']) + 1e-5)
    }])
    
    # Predict Target(t) which represents NetCash(t+1)
    predicted_val = float(model.predict(pred_features[feature_cols])[0])
    
    # Display KPIs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📅 Bulan Data Terakhir</div>
            <div class="metric-value">{latest_row['month']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📥 Kas Bersih Terakhir</div>
            <div class="metric-value" style="color: {'#2e7d32' if latest_row['net_cash'] >= 0 else '#c62828'}">
                Rp {latest_row['net_cash']:,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #e74c3c;">
            <div class="metric-label">🔮 Proyeksi Bulan Depan ({next_month_str})</div>
            <div class="metric-value" style="color: {'#2e7d32' if predicted_val >= 0 else '#c62828'}">
                Rp {predicted_val:,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Layout Main Panel: Forecast and Plot
    tab1, tab2 = st.tabs(["📊 Tren & Proyeksi Bulanan", "📋 Tabel Fitur Detail"])
    
    with tab1:
        st.subheader("Kurva Perbandingan Aktual vs Proyeksi Arus Kas Bersih")
        
        # Prepare plot data
        # Historical actual values (all valid months)
        plot_df = features_df.dropna(subset=['net_cash']).copy()
        
        fig = go.Figure()
        
        # Plot historical net_cash
        fig.add_trace(go.Scatter(
            x=plot_df['month'],
            y=plot_df['net_cash'],
            mode='lines+markers',
            name='Net Cashflow Historis',
            line=dict(color='#2c5364', width=3),
            marker=dict(size=8)
        ))
        
        # Connect last actual month with the next predicted month
        connect_months = [plot_df['month'].iloc[-1], next_month_str]
        connect_values = [plot_df['net_cash'].iloc[-1], predicted_val]
        
        # Plot forecast point
        fig.add_trace(go.Scatter(
            x=connect_months,
            y=connect_values,
            mode='lines+markers',
            name='Proyeksi Bulan Depan (XGBoost)',
            line=dict(color='#e74c3c', width=3, dash='dash'),
            marker=dict(size=8, symbol='x')
        ))
        
        fig.update_layout(
            title="Tren Bulanan Arus Kas Bersih Koperasi (Rupiah)",
            xaxis_title="Bulan",
            yaxis_title="Nominal (Rp)",
            legend_title="Kategori",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Dataset Fitur Bulanan Hasil Rekayasa")
        
        # Format table for display
        display_df = features_df.copy()
        for col in ['cash_in', 'cash_out', 'net_cash', 'lag_1', 'lag_2', 'rolling_mean_3', 'target']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"Rp {x:,.2f}" if pd.notna(x) else "-")
        display_df['growth_rate'] = display_df['growth_rate'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")
        
        st.dataframe(display_df, use_container_width=True)
        st.info("💡 **Catatan**: Target pada baris terakhir bernilai '-' (kosong) karena merupakan variabel target aktual dari periode berikutnya yang akan diprediksi oleh model.")

else:
    st.info("💡 Hubungkan dengan data bersih `clean_transactions.csv` untuk memulai. Silakan unggah berkas di panel sebelah kiri.")
