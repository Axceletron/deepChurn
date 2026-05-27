import streamlit as st

st.title("Customer Churn Prediction")

st.write(
    "Deep Learning ANN Project"
)

import joblib
import pandas as pd

from tensorflow.keras.models import load_model

model = load_model(
    '../models/churn_ann_model.keras'
)
scaler = joblib.load(
    '../models/scaler.pkl'
)
feature_columns = joblib.load(
    '../models/feature_columns.pkl'
)
tenure = st.slider(
    'Tenure',
    0,
    72,
    12
)

monthly_charges = st.number_input(
    'Monthly Charges',
    0.0,
    200.0,
    70.0
)
gender = st.selectbox(
    'Gender',
    ['Male', 'Female']
)

contract = st.selectbox(
    'Contract Type',
    [
        'Month-to-month',
        'One year',
        'Two year'
    ]
)
if st.button("Predict Churn"):
    input_data = {
    'gender': gender,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'Contract': contract
    }
        
    input_data = {
        'gender': gender,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'Contract': contract
    }

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )
    input_scaled = scaler.transform(
        input_df
    )
    prediction_prob = model.predict(
        input_scaled
    )
    prediction = (
        prediction_prob > 0.5
    ).astype(int)
    st.subheader("Prediction Result")
    st.write(
        f"Churn Probability: "
        f"{prediction_prob[0][0]*100:.2f}%"
    )
    if prediction_prob > 0.5:

        st.error(
            "Customer likely to churn"
        )

    else:

        st.success(
            "Customer likely to stay"
        )
