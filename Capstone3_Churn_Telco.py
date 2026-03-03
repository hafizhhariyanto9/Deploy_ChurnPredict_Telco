import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("📊 Telco Customer Churn Predictor")
st.markdown("Aplikasi ini memprediksi kemungkinan seorang pelanggan akan churn (berhenti berlangganan) berdasarkan data historis mereka. Gunakan aplikasi ini untuk mengidentifikasi pelanggan berisiko tinggi dan mengambil tindakan retensi yang tepat.")
st.markdown("---")

# Cek file model
@st.cache_resource
def check_model_files():
    required_files = ['best_pipeline.pkl', 'preprocessor.pkl', 'model_metadata.json']
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    return missing_files

missing_files = check_model_files()
if missing_files:
    st.error("File model tidak ditemukan! Pastikan file berikut ada:")
    for file in missing_files:
        st.code(file)
    st.stop()

# Load model
@st.cache_resource
def load_model():
    try:
        with open('best_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return pipeline, preprocessor, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

pipeline, preprocessor, metadata = load_model()

if pipeline is None or preprocessor is None:
    st.stop()

# Ambil informasi metadata
if metadata is not None:
    threshold = metadata.get('best_threshold', 0.25)
    model_name = metadata.get('best_model_name', 'Logistic Regression')
else:
    threshold = 0.25
    model_name = 'Logistic Regression'

st.success(f"✅ Model siap digunakan | Threshold: {threshold:.0%}")
st.markdown("---")

# Sidebar input
st.sidebar.header("📝 Input Data Pelanggan")

def get_user_input():
    # Informasi Dasar
    st.sidebar.subheader("Informasi Dasar")
    tenure = st.sidebar.slider('Lama Berlangganan (bulan)', 0, 72, 12)
    monthly_charges = st.sidebar.number_input('Biaya Bulanan (USD)', 18.0, 120.0, 65.0, step=5.0)
    
    # Demografi
    st.sidebar.subheader("Demografi")
    dependents = st.sidebar.radio('Memiliki Tanggungan', ['Yes', 'No'], horizontal=True)
    
    # Layanan
    st.sidebar.subheader("Layanan")
    internet_service = st.sidebar.selectbox('Jenis Internet', ['DSL', 'Fiber optic', 'No'])
    
    if internet_service != 'No':
        online_security = st.sidebar.radio('Online Security', ['Yes', 'No'], horizontal=True, key='os')
        online_backup = st.sidebar.radio('Online Backup', ['Yes', 'No'], horizontal=True, key='ob')
        device_protection = st.sidebar.radio('Device Protection', ['Yes', 'No'], horizontal=True, key='dp')
        tech_support = st.sidebar.radio('Tech Support', ['Yes', 'No'], horizontal=True, key='ts')
    else:
        online_security = online_backup = device_protection = tech_support = 'No internet service'
    
    # Kontrak
    st.sidebar.subheader("Kontrak & Billing")
    contract = st.sidebar.selectbox('Jenis Kontrak', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.sidebar.radio('Paperless Billing', ['Yes', 'No'], horizontal=True)
    
    # Feature engineering
    if tenure <= 12:
        tenure_category = 'New (0-12m)'
    elif tenure <= 24:
        tenure_category = 'Medium (13-24m)'
    elif tenure <= 48:
        tenure_category = 'Long (25-48m)'
    else:
        tenure_category = 'Loyal (49-72m)'
    
    services = [online_security, online_backup, device_protection, tech_support]
    total_services = sum(1 for s in services if s == 'Yes')
    has_internet = 1 if internet_service != 'No' else 0
    
    return pd.DataFrame([{
        'Dependents': dependents, 'tenure': tenure, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'InternetService': internet_service,
        'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'Contract': contract, 'PaperlessBilling': paperless_billing,
        'MonthlyCharges': monthly_charges, 'tenure_category': tenure_category,
        'total_services': total_services, 'has_internet': has_internet
    }])

df_input = get_user_input()

# Layout utama
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Data Pelanggan")
    
    # Dataframe untuk tampilan
    display_data = {
        'Lama Berlangganan': f"{df_input['tenure'].values[0]} bulan",
        'Biaya Bulanan': f"${df_input['MonthlyCharges'].values[0]:.2f}",
        'Jenis Kontrak': df_input['Contract'].values[0],
        'Layanan Internet': df_input['InternetService'].values[0],
        'Memiliki Tanggungan': df_input['Dependents'].values[0],
        'Paperless Billing': df_input['PaperlessBilling'].values[0],
        'Online Security': df_input['OnlineSecurity'].values[0],
        'Online Backup': df_input['OnlineBackup'].values[0],
        'Device Protection': df_input['DeviceProtection'].values[0],
        'Tech Support': df_input['TechSupport'].values[0],
        'Kategori Tenure': df_input['tenure_category'].values[0],
        'Total Layanan Tambahan': df_input['total_services'].values[0],
        'Berlangganan Internet': 'Ya' if df_input['has_internet'].values[0] == 1 else 'Tidak'
    }
    
    # Tampilkan sebagai dataframe
    df_display = pd.DataFrame(list(display_data.items()), columns=['Fitur', 'Nilai'])
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    st.info(f"**Model:** {model_name}")

with col2:
    st.subheader("🔮 Hasil Prediksi")
    
    try:
        # Prediksi
        data_proc = preprocessor.transform(df_input)
        prob = pipeline.predict_proba(data_proc)[0][1]
        pred = 1 if prob >= threshold else 0
        
        # Tampilkan probabilitas
        st.write(f"**Probabilitas Churn:** {prob:.1%}")
        
        # Status
        if pred == 1:
            st.write(f"**Status:** BERISIKO CHURN (di atas {threshold:.0%})")
            
            # Tingkat risiko
            if prob >= 0.75:
                st.write("**Tingkat Risiko:** Tinggi")
            elif prob >= 0.5:
                st.write("**Tingkat Risiko:** Sedang")
            else:
                st.write("**Tingkat Risiko:** Rendah")
        else:
            st.write(f"**Status:** AMAN (di bawah {threshold:.0%})")
            st.write("**Tingkat Risiko:** Sangat Rendah")
        
        # Progress bar sederhana
        st.progress(float(prob))
        st.caption(f"Threshold: {threshold:.0%}")
        
        st.markdown("---")
        
        # Rekomendasi
        st.subheader("📌 Rekomendasi Tindakan")
        
        if prob >= 0.75:
            st.write("**Prioritas Tinggi - Intervensi Segera**")
            st.write("- Hubungi pelanggan via telepon")
            st.write("- Tawarkan diskon kontrak tahunan")
        elif prob >= 0.5:
            st.write("**Prioritas Sedang - Tindakan Preventif**")
            st.write("- Kirim email personal")
            st.write("- Tawarkan uji coba gratis layanan")
        elif prob >= 0.25:
            st.write("**Prioritas Rendah - Monitor**")
            st.write("- Masukkan ke program loyalty")
            st.write("- Kirim newsletter")
        else:
            st.write("**Pelanggan Stabil**")
            st.write("- Program loyalty standar")
            st.write("- Minta testimonial")
            
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("© 2026 Telco Churn Predictor")