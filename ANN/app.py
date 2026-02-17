import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1f4037, #99f2c8);
    }
    .title {
        font-size: 45px;
        font-weight: bold;
        text-align: center;
        color: white;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        color: #155724;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    .danger-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        color: #721c24;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ---------------- HEADER ---------------- #
st.markdown('<div class="title">🏦 Customer Churn Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict whether a customer will leave the bank using AI</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- LAYOUT ---------------- #
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("### 🧾 Customer Information")

    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 30)
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900, value=600)
    tenure = st.slider('📅 Tenure (Years)', 0, 10, 5)

with col2:
    st.markdown("### 💰 Financial Details")

    balance = st.number_input('🏦 Balance', min_value=0.0, value=50000.0)
    estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, value=50000.0)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
    is_active_member = st.selectbox('🔥 Is Active Member?', [0, 1])

st.markdown("---")

# ---------------- PREDICTION BUTTON ---------------- #
if st.button("🚀 Predict Churn"):

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.markdown("## 📊 Prediction Result")

    st.metric("Churn Probability", f"{prediction_proba*100:.2f}%")

    if prediction_proba > 0.5:
        st.markdown(
            '<div class="danger-box">⚠️ The customer is likely to churn.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="success-box">✅ The customer is not likely to churn.</div>',
            unsafe_allow_html=True
        )

    # Progress bar visualization
    st.progress(float(prediction_proba))

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & TensorFlow")
