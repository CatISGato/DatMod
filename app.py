import streamlit as st
import joblib
import pandas as pd

# Load model dan fitur
model = joblib.load('xgb_model_final.pkl')
features = joblib.load('fitur_model.pkl')

st.title("Prediksi Customer Churn")
st.write("Masukkan data pelanggan untuk melihat apakah pelanggan akan churn atau tidak.")

# Membuat form input
with st.form("input_form"):
    tenure = st.number_input("Tenure", min_value=0.0)
    warehouse_to_home = st.number_input("Warehouse to Home", min_value=0.0)
    satisfaction_score = st.slider("Satisfaction Score", 1, 5)
    complain = st.selectbox("Complain (0: Tidak, 1: Ya)", [0, 1])
    
    submit = st.form_submit_button("Prediksi")

if submit:
    # Memasukkan input ke DataFrame sesuai urutan fitur
    input_data = pd.DataFrame([[tenure, warehouse_to_home, satisfaction_score, complain]], 
                              columns=['Tenure', 'WarehouseToHome', 'SatisfactionScore', 'Complain'])
    
    # Melakukan prediksi
    prediction = model.predict(input_data)
    result = "Churn" if prediction[0] == 1 else "Tidak Churn"
    
    st.subheader(f"Hasil Prediksi: {result}")
