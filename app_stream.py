import streamlit as st
import pandas as pd
import joblib
import numpy as np

# biar ga error kalo datanya kebanyakan
pd.set_option("styler.render.max_elements", 99999999)

# setting halaman
st.set_page_config(page_title="E-Wallet Guard Enterprise v5.0", page_icon="🛡️", layout="wide")

def load_css():
    with open("style.css", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# load model yang sudah ditrain sebelumnya
@st.cache_resource
def load_model():
    try:
        return joblib.load('qris_fraud_model.pkl')
    except Exception as e:
        st.error("Model 'qris_fraud_model.pkl' tidak ditemukan! Jalankan train_model_v4.py terlebih dahulu.")
        st.stop()

model = load_model()
features_list = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_TRANSFER', 'Frekuensi_Transaksi', 'Jam_Aktual', 'Is_Jam_Malam', 'Anomali_Uang_Tujuan']

# bagian header paling atas
st.markdown("""
<div class="hero-header">
    <div class="hero-logo-wrap">
        <div class="hero-logo-ring-outer"></div>
        <div class="hero-logo-ring"></div>
        <div class="hero-logo">🛡️</div>
    </div>
    <div class="hero-badge">Enterprise Monitoring Platform</div>
    <h1 class="hero-title">E-Wallet Guard</h1>
    <ul class="hero-list">
        <li>E-Wallet Guard v5.0</li>
        <li>Fraud Detection Dashboard</li>
        <li>Real-Time E-Wallet Monitoring System</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-intro">
    <h3>Ekosistem Pengawasan Finansial & Deteksi Fraud Transaksi E-Wallet Kampus</h3>
    <p class="caption-text">Dikembangkan sesuai spesifikasi teknis Analitika Keuangan dan Machine Learning Fintech.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# bikin tab, ada 3 bagian utama
tab1, tab2, tab3 = st.tabs(["Simulator Transaksi", "Audit Transaksi Masal", "Performa & Metrik Model"])

# TAB 1 - buat cek transaksi satu-satu
with tab1:
    st.markdown("""
    <div class="tab-hero">
        <div class="tab-hero-icon">📱</div>
        <div>
            <h2 class="tab-hero-title">Verifikasi Transaksi Tunggal</h2>
            <p class="tab-hero-desc">Gunakan simulator ini saat melayani mahasiswa di meja kasir secara langsung.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="col-panel-marker"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="form-panel form-panel-buyer">
            <div class="panel-header">
                <span class="panel-icon">🛒</span>
                <div>
                    <div class="panel-title">Sisi Aplikasi Pembeli</div>
                    <div class="panel-subtitle">Data transaksi dari perangkat mahasiswa</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        waktu = st.slider("Jam Transaksi (0–23)", 0, 23, 14)
        frekuensi = st.number_input("Frekuensi Scan HP (1 Jam Terakhir)", min_value=1, value=1)
        nominal = st.number_input("Nominal pada Struk (Rp)", min_value=0, value=50000, step=5000)
        tipe = st.selectbox("Tipe Transaksi Sistem", ["TRANSFER", "PAYMENT"])

    with col2:
        st.markdown('<div class="col-panel-marker"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="form-panel form-panel-bank">
            <div class="panel-header">
                <span class="panel-icon">🏦</span>
                <div>
                    <div class="panel-title">Sisi Validasi Saldo Bank</div>
                    <div class="panel-subtitle">Mutasi rekening pengirim & merchant kantin</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        saldo_awal_p = st.number_input("Saldo Awal Akun Pengirim (Rp)", min_value=0, value=150000, step=10000)
        saldo_akhir_p = st.number_input("Saldo Akhir Akun Pengirim (Rp)", min_value=0, value=100000, step=10000)
        saldo_awal_m = st.number_input("Saldo Awal Rekening Kantin (Rp)", min_value=0, value=250000)
        saldo_akhir_m = st.number_input("Saldo Akhir Rekening Kantin (Rp)", min_value=0, value=250000)

    st.markdown('<div class="verify-btn-wrap">', unsafe_allow_html=True)
    if st.button("🔍  VERIFIKASI TRANSAKSI KASIR", type="primary", use_container_width=True):
        # ngitung fitur-fitur yg dibutuhin model
        tipe_transfer = 1 if tipe == "TRANSFER" else 0
        jam_aktual = waktu % 24
        is_jam_malam = 1 if jam_aktual <= 5 else 0
        uang_diterima = saldo_akhir_m - saldo_awal_m
        anomali_uang_tujuan = nominal - uang_diterima

        data_input = pd.DataFrame([{f: 0 for f in features_list}])
        data_input['step'] = waktu
        data_input['amount'] = nominal
        data_input['oldbalanceOrg'] = saldo_awal_p
        data_input['newbalanceOrig'] = saldo_akhir_p
        data_input['oldbalanceDest'] = saldo_awal_m
        data_input['newbalanceDest'] = saldo_akhir_m
        data_input['type_TRANSFER'] = tipe_transfer
        data_input['Frekuensi_Transaksi'] = frekuensi
        data_input['Jam_Aktual'] = jam_aktual
        data_input['Is_Jam_Malam'] = is_jam_malam
        data_input['Anomali_Uang_Tujuan'] = anomali_uang_tujuan

        data_input = data_input[features_list]

        prediksi = model.predict(data_input)[0]
        probabilitas = model.predict_proba(data_input)[0][1] * 100

        st.markdown("#### 📋 Hasil Analisis Risiko AI")

        if prediksi == 1:
            st.markdown(f"""
            <div class="result-card danger">
                <div class="result-icon">🚨</div>
                <div class="result-title">TRANSAKSI DITOLAK</div>
                <div class="risk-gauge-wrap">
                    <div class="risk-gauge danger" style="--val: {probabilitas};">
                        <div class="risk-gauge-inner">
                            <div class="risk-gauge-value">{probabilitas:.1f}%</div>
                            <div class="risk-gauge-label">Skor Risiko</div>
                        </div>
                    </div>
                </div>
                <div class="result-recommendation">
                    <strong>Rekomendasi Bisnis:</strong> Terdeteksi anomali finansial fatal (Uang tidak masuk ke rekening tujuan namun struk tertera sukses).
                    <br><br>
                    <strong>TINDAKAN:</strong> Jangan berikan makanan, minta pembeli menunjukkan mutasi keluar dari aplikasi aslinya.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card safe">
                <div class="result-icon">✅</div>
                <div class="result-title">TRANSAKSI AMAN</div>
                <div class="risk-gauge-wrap">
                    <div class="risk-gauge safe" style="--val: {probabilitas};">
                        <div class="risk-gauge-inner">
                            <div class="risk-gauge-value">{probabilitas:.1f}%</div>
                            <div class="risk-gauge-label">Skor Risiko</div>
                        </div>
                    </div>
                </div>
                <div class="result-recommendation">
                    <strong>Rekomendasi Bisnis:</strong> Pola transaksi normal dan mutasi perbankan sinkron secara valid.
                    <br><br>
                    Pesanan aman diberikan kepada mahasiswa.
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2 - upload csv buat cek banyak transaksi sekaligus
with tab2:
    st.markdown("""
    <div class="tab-hero">
        <div class="tab-hero-icon">📊</div>
        <div>
            <h2 class="tab-hero-title">Otomatisasi Audit Harian</h2>
            <p class="tab-hero-desc">Unggah riwayat transaksi harian kantin (.csv) untuk mendeteksi penipuan massal secara instan.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Unggah File Rekam Transaksi Kantin (.csv)", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_data = pd.read_csv(uploaded_file)
            st.success("✅ File berhasil diunggah! Memulai pemrosesan otomatis...")

            processed_df = raw_data.copy()
            if 'type' in processed_df.columns:
                processed_df = processed_df[(processed_df['type'] == 'PAYMENT') | (processed_df['type'] == 'TRANSFER')]
                processed_df['type_TRANSFER'] = processed_df['type'].apply(lambda x: 1 if x == 'TRANSFER' else 0)
            else:
                processed_df['type_TRANSFER'] = 0

            # feature engineering kayak di training
            processed_df['Frekuensi_Transaksi'] = processed_df.groupby(['nameOrig', 'step'])['amount'].transform('count')
            processed_df['Jam_Aktual'] = processed_df['step'] % 24
            processed_df['Is_Jam_Malam'] = processed_df['Jam_Aktual'].apply(lambda x: 1 if x <= 5 else 0)
            processed_df['Uang_Diterima'] = processed_df['newbalanceDest'] - processed_df['oldbalanceDest']
            processed_df['Anomali_Uang_Tujuan'] = processed_df['amount'] - processed_df['Uang_Diterima']

            X_batch = processed_df[features_list]

            # prediksi pake model
            processed_df['Fraud_Prediction'] = model.predict(X_batch)
            processed_df['Risk_Score (%)'] = model.predict_proba(X_batch)[:, 1] * 100

            total_tx = len(processed_df)
            total_fraud = int(processed_df['Fraud_Prediction'].sum())
            total_loss_prevented = processed_df[processed_df['Fraud_Prediction'] == 1]['amount'].sum()

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card info">
                    <div class="metric-card-icon">📋</div>
                    <div class="metric-card-label">Total Transaksi Diaudit</div>
                    <div class="metric-card-value">{total_tx:,}</div>
                </div>
                <div class="metric-card danger">
                    <div class="metric-card-icon">🚨</div>
                    <div class="metric-card-label">Penipuan Terdeteksi</div>
                    <div class="metric-card-value">{total_fraud} Kasus</div>
                </div>
                <div class="metric-card success">
                    <div class="metric-card-icon">💰</div>
                    <div class="metric-card-label">Kerugian Diselamatkan</div>
                    <div class="metric-card-value">Rp {total_loss_prevented:,.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("📑 Daftar Transaksi Mencurigakan Terfilter (Sinyal Merah) - Top 1000 Teratas")
            fraud_only = processed_df[processed_df['Fraud_Prediction'] == 1][['step', 'amount', 'nameOrig', 'nameDest', 'Risk_Score (%)']]
            if not fraud_only.empty:
                # sort dari risk score paling tinggi dulu
                fraud_display = fraud_only.sort_values(by=['Risk_Score (%)', 'step'], ascending=[False, True]).head(1000)
                st.dataframe(fraud_display.style.background_gradient(subset=['Risk_Score (%)'], cmap='Reds'), use_container_width=True)
            else:
                st.success("🎉 Luar biasa! Tidak ada indikasi fraud terdeteksi dari data yang diunggah hari ini.")

        except Exception as e:
            st.error(f"Format kolom file CSV tidak sesuai dengan kebutuhan model AI. Error: {str(e)}")

# TAB 3 - info performa model
with tab3:
    st.markdown("""
    <div class="tab-hero">
        <div class="tab-hero-icon">⚙️</div>
        <div>
            <h2 class="tab-hero-title">Transparansi & Evaluasi Model</h2>
            <p class="tab-hero-desc">Validitas pengujian model Random Forest dengan teknik balancing SMOTE.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div class="eval-card">
            <div class="card-title">Classification Report</div>
            <p style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 0;">
                Berdasarkan hasil pengujian data testing menggunakan teknik balancing data <strong style="color: var(--text-primary);">SMOTE</strong>:
            </p>
            <div class="eval-metric-grid">
                <div class="eval-metric-item">
                    <div class="value">96%</div>
                    <div class="label">Akurasi</div>
                </div>
                <div class="eval-metric-item">
                    <div class="value">99%</div>
                    <div class="label">Precision</div>
                </div>
                <div class="eval-metric-item">
                    <div class="value">96%</div>
                    <div class="label">Recall</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="eval-card">
            <div class="card-title">Informasi Model</div>
            <p style="color: var(--text-secondary); font-size: 0.875rem; line-height: 1.8;">
                <strong style="color: var(--accent-emerald-light);">Algoritma:</strong> Random Forest Classifier<br>
                <strong style="color: var(--accent-emerald-light);">Balancing:</strong> SMOTE Oversampling<br>
                <strong style="color: var(--accent-emerald-light);">Fitur:</strong> 11 variabel engineered<br>
                <strong style="color: var(--accent-emerald-light);">Indikator Kunci:</strong> Anomali_Uang_Tujuan
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### **Analisis Signifikansi Fitur (Feature Importance)**")
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=features_list).sort_values(ascending=True)
    st.bar_chart(feat_importances, color="#10b981")
    st.caption("Grafik di atas membuktikan bahwa fitur buatan kelompok kami (*Anomali_Uang_Tujuan*) adalah indikator paling fatal yang digunakan otak AI untuk mendeteksi penipuan QRIS.")