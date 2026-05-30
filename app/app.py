import streamlit as st
from utils import (
    prediction_churn,
    preprocess_input
)
from app.explainability_1 import generate_shap_plot

# =========================
# Page Title
# =========================

st.title(
    "Customer Churn Prediction"
)

st.write(
    "Deep Learning ANN Project"
)


# =========================
# User Inputs
# =========================

gender = st.selectbox(
    'Gender',
    ['Male', 'Female']
)

senior_citizen = st.selectbox(
    'Senior Citizen',
    [0, 1]
)

partner = st.selectbox(
    'Partner',
    ['Yes', 'No']
)

dependents = st.selectbox(
    'Dependents',
    ['Yes', 'No']
)

tenure = st.slider(
    'Tenure',
    0,
    72,
    12
)

phone_service = st.selectbox(
    'Phone Service',
    ['Yes', 'No']
)

internet_service = st.selectbox(
    'Internet Service',
    [
        'DSL',
        'Fiber optic',
        'No'
    ]
)

contract = st.selectbox(
    'Contract Type',
    [
        'Month-to-month',
        'One year',
        'Two year'
    ]
)

monthly_charges = st.number_input(
    'Monthly Charges',
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total_charges = st.number_input(
    'Total Charges',
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)


# =========================
# Prediction Button
# =========================

if st.button("Reasoned Predict Churn"):

    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'InternetService': internet_service,
        'Contract': contract,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    prediction, probability = prediction_churn(input_data)

    processed_data = preprocess_input(input_data)
    # =====================
    # Results
    # =====================

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    if prediction == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")

    st.subheader("Model Explainability")
    shap_plot = generate_shap_plot(processed_data)
    st.pyplot(shap_plot)

elif st.button("Predict Churn"):

    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'InternetService': internet_service,
        'Contract': contract,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    prediction, probability = prediction_churn(input_data)
    # =====================
    # Results
    # =====================

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    if prediction == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")