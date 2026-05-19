import streamlit as st
import joblib
import pandas as pd

# Load model dan daftar fitur yang digunakan saat training
model = joblib.load('xgb_model_final.pkl')
features = joblib.load('fitur_model.pkl')

st.title("Prediksi Customer Churn")
st.write("Silakan masukkan data pelanggan di bawah ini untuk melihat hasil prediksi.")

# Membuat form input dinamis berdasarkan fitur yang ada di model
with st.form("input_form"):
    user_inputs = {}
    
    # Membuat input field secara otomatis berdasarkan list fitur
    for feature in features:
        # Pengecualian: Sesuaikan tipe input berdasarkan nama kolom jika perlu
        if feature in ['Tenure', 'WarehouseToHome', 'SatisfactionScore', 'CouponUsed', 'OrderAmountHikeFromlastYear']:
            user_inputs[feature] = st.number_input(f"{feature}", min_value=0.0)
        else:
            user_inputs[feature] = st.number_input(f"{feature}")

    submit = st.form_submit_button("Prediksi Sekarang")

if submit:
    # Mengonversi input ke DataFrame
    input_data = pd.DataFrame([user_inputs])
    
    # Melakukan prediksi
    prediction = model.predict(input_data)
    result = "Churn (Berhenti Berlangganan)" if prediction[0] == 1 else "Tidak Churn (Tetap Setia)"
    
    st.subheader(f"Hasil Prediksi: {result}")
    
    # Menampilkan data yang dimasukkan
    st.write("Data yang dianalisis:", input_data)
