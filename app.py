import streamlit as st
import pandas as pd
import joblib

# 1. Load Model dan Daftar Kolom Fitur
try:
    model = joblib.load('xgb_model_final.pkl')
    kolom_fitur = joblib.load('fitur_model.pkl')
except FileNotFoundError:
    st.error("⚠️ File model (.pkl) tidak ditemukan! Pastikan file berada di folder yang sama.")
    st.stop()

# 2. Pengaturan UI
st.set_page_config(page_title="E-Commerce Churn Prediction", page_icon="🛒", layout="centered")
st.title("🛒 Prediksi Churn Pelanggan")
st.write("Aplikasi prediksi apakah pelanggan berpotensi berhenti berlangganan (Churn).")
st.markdown("---")

# 3. Form Input
col1, col2 = st.columns(2)
with col1:
    tenure = st.number_input("Masa Berlangganan (Tenure)", min_value=0, value=5)
    warehouse_to_home = st.number_input("Jarak Gudang ke Rumah (KM)", min_value=0, value=10)
    hour_spend_on_app = st.number_input("Jam di Aplikasi", min_value=0.0, value=3.0)
    number_of_device = st.number_input("Jumlah Perangkat", min_value=1, value=3)
    satisfaction_score = st.slider("Skor Kepuasan (1-5)", 1, 5, 3)
    complain = st.selectbox("Pernah Komplain?", ["Tidak", "Ya"])

with col2:
    number_of_address = st.number_input("Jumlah Alamat", min_value=1, value=2)
    order_amount_hike = st.number_input("Kenaikan Nilai Pesanan (%)", min_value=0, value=12)
    coupon_used = st.number_input("Jumlah Kupon", min_value=0, value=1)
    order_count = st.number_input("Total Pesanan", min_value=1, value=2)
    day_since_last_order = st.number_input("Hari Sejak Pesanan Terakhir", min_value=0, value=4)
    cashback_amount = st.number_input("Rata-rata Cashback ($)", min_value=0.0, value=150.0)

customer_segment = st.selectbox("Segmen Pelanggan", 
    ["0 - New Shopper", "1 - VIP/Heavy Shopper", "2 - Dormant Customer"])

# 4. Prediksi
if st.button("🚀 Analisis Risiko Churn", use_container_width=True):
    # Inisialisasi dictionary dengan nilai 0 untuk semua fitur
    input_data = {col: 0 for col in kolom_fitur}
    
    # Mapping input pengguna ke dictionary
    mapping = {
        'Tenure': tenure,
        'WarehouseToHome': warehouse_to_home,
        'HourSpendOnApp': hour_spend_on_app,
        'NumberOfDeviceRegistered': number_of_device,
        'SatisfactionScore': satisfaction_score,
        'NumberOfAddress': number_of_address,
        'Complain': 1 if complain == "Ya" else 0,
        'OrderAmountHikeFromlastYear': order_amount_hike,
        'CouponUsed': coupon_used,
        'OrderCount': order_count,
        'DaySinceLastOrder': day_since_last_order,
        'CashbackAmount': cashback_amount,
        'Customer_Segment': int(customer_segment[0])
    }
    
    # Update hanya fitur yang ada di model
    for key, value in mapping.items():
        if key in input_data:
            input_data[key] = value
            
    df_prediction = pd.DataFrame([input_data])
    
    # Prediksi
    prediction = model.predict(df_prediction)
    probability = model.predict_proba(df_prediction)[0][1]
    
    # 5. Output
    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"⚠️ **PERINGATAN:** Pelanggan diprediksi **CHURN**. (Risiko: {probability:.2%})")
    else:
        st.success(f"✅ **KONDISI AMAN:** Pelanggan diprediksi **RETAINED**. (Risiko: {probability:.2%})")
