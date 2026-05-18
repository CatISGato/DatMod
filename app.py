import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load Model dan Daftar Kolom Fitur
try:
    model = joblib.load('xgb_model_final.pkl')
    kolom_fitur = joblib.load('fitur_model.pkl')
except FileNotFoundError:
    st.error("⚠️ File model (.pkl) tidak ditemukan! Pastikan Anda sudah menjalankan seluruh Cell di Jupyter Notebook (terutama Cell ekspor model) sehingga file 'xgb_model_final.pkl' dan 'fitur_model.pkl' terbentuk di folder yang sama.")
    st.stop()

# 2. Pengaturan Konfigurasi Halaman UI
st.set_page_config(page_title="E-Commerce Churn Prediction", page_icon="🛒", layout="centered")

st.title("🛒 Prediksi Churn Pelanggan E-Commerce")
st.write("Aplikasi ini menggunakan model Machine Learning (XGBoost) yang dikembangkan oleh **Kelompok 11** untuk memprediksi apakah seorang pelanggan berpotensi berhenti menggunakan layanan (Churn).")
st.markdown("---")

# 3. Pembuatan Form Input Data Pelanggan
st.subheader("📊 Masukkan Data Transaksi & Perilaku Pelanggan")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Masa Berlangganan (Tenure - Bulan)", min_value=0, max_value=120, value=5)
    warehouse_to_home = st.number_input("Jarak Gudang ke Rumah (KM)", min_value=0, max_value=200, value=10)
    hour_spend_on_app = st.number_input("Jam yang Dihabiskan di Aplikasi", min_value=0.0, max_value=24.0, value=3.0)
    number_of_device = st.number_input("Jumlah Perangkat Terdaftar", min_value=1, max_value=10, value=3)
    satisfaction_score = st.slider("Skor Kepuasan Pelanggan (1-5)", min_value=1, max_value=5, value=3)
    complain = st.selectbox("Apakah Pernah Komplain dalam Bulan Terakhir?", ["Tidak", "Ya"])

with col2:
    number_of_address = st.number_input("Jumlah Alamat Terdaftar", min_value=1, max_value=50, value=2)
    order_amount_hike = st.number_input("Kenaikan Persentase Nilai Pesanan (%)", min_value=0, max_value=100, value=12)
    coupon_used = st.number_input("Jumlah Kupon yang Digunakan", min_value=0, max_value=50, value=1)
    order_count = st.number_input("Total Jumlah Pesanan", min_value=1, max_value=100, value=2)
    day_since_last_order = st.number_input("Hari Sejak Pesanan Terakhir", min_value=0, max_value=365, value=4)
    cashback_amount = st.number_input("Rata-rata Nilai Cashback ($)", min_value=0.0, max_value=1000.0, value=150.0)

st.markdown("---")
st.subheader("🤖 Hasil Integrasi Fitur Lanjutan")
customer_segment = st.selectbox("Kategori Segmen Pelanggan (Hasil Analisis K-Means Clustering)", 
                                  ["0 - Pelanggan Baru / Aktivitas Rendah (Recent Shoppers)", 
                                   "1 - Pelanggan Setia & Pembelanja Tinggi (VIP/Heavy Shoppers)", 
                                   "2 - Pelanggan Pasif / Berisiko Tinggi (Dormant Customer)"])

# 4. Proses Eksekusi Prediksi Saat Tombol Diklik
if st.button("🚀 Analisis Risiko Churn Pelanggan", use_container_width=True):
    
    # Membuat dictionary kosong dengan struktur tepat sesuai fitur_model.pkl (semua bernilai default 0)
    input_data = {col: 0 for col in kolom_fitur}
    
    # Mengisi nilai ke dalam dictionary berdasarkan input form
    if 'Tenure' in input_data: input_data['Tenure'] = tenure
    if 'WarehouseToHome' in input_data: input_data['WarehouseToHome'] = warehouse_to_home
    if 'HourSpendOnApp' in input_data: input_data['HourSpendOnApp'] = hour_spend_on_app
    if 'NumberOfDeviceRegistered' in input_data: input_data['NumberOfDeviceRegistered'] = number_of_device
    if 'SatisfactionScore' in input_data: input_data['SatisfactionScore'] = satisfaction_score
    if 'NumberOfAddress' in input_data: input_data['NumberOfAddress'] = number_of_address
    if 'Complain' in input_data: input_data['Complain'] = 1 if complain == "Ya" else 0
    if 'OrderAmountHikeFromlastYear' in input_data: input_data['OrderAmountHikeFromlastYear'] = order_amount_hike
    if 'CouponUsed' in input_data: input_data['CouponUsed'] = coupon_used
    if 'OrderCount' in input_data: input_data['OrderCount'] = order_count
    if 'DaySinceLastOrder' in input_data: input_data['DaySinceLastOrder'] = day_since_last_order
    if 'CashbackAmount' in input_data: input_data['CashbackAmount'] = cashbackamount = cashback
    if 'Customer_Segment' in input_data: input_data['Customer_Segment'] = int(customer_segment[0])
    
    # Mengonversi dictionary menjadi DataFrame Pandas tunggal
    df_prediction = pd.DataFrame([input_data])
    
    # Melakukan kalkulasi prediksi menggunakan model XGBoost
    prediction = model.predict(df_prediction)
    probability = model.predict_proba(df_prediction)[0][1]
    
    # 5. Menampilkan Hasil Output Prediksi Secara Interaktif
    st.markdown("---")
    st.subheader("📊 Hasil Kesimpulan Analisis:")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **PERINGATAN KRITIS:** Pelanggan diprediksi akan **CHURN** (Berhenti Berlangganan).")
        st.metric(label="Probabilitas Risiko Churn", value=f"{probability:.2%}")
        st.info("💡 **Rekomendasi Strategi Retensi:** Segera berikan penawaran insentif khusus, kupon diskon personal, atau hubungi pelanggan melalui tim customer care untuk menangani keluhan potensial.")
    else:
        st.success(f"✅ **KONDISI AMAN:** Pelanggan diprediksi akan **RETAINED** (Tetap Setia Berlangganan).")
        st.metric(label="Probabilitas Risiko Churn", value=f"{probability:.2%}")
        st.caption("Pelanggan menunjukkan loyalitas yang baik berdasarkan indikator transaksi mereka saat ini. Tetap pertahankan kualitas layanan.")
