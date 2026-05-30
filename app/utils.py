import joblib
import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR,'models','churn_ann_model.keras')
SCALER_PATH=os.path.join(BASE_DIR,'models','scaler.pkl')
FEATURE_PATH=os.path.join(BASE_DIR,'models','feature_columns.pkl')
BACKGROUND_PATH = os.path.join(BASE_DIR,'data/processed','X_train.npy')

model = load_model(MODEL_PATH)
print(model)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_PATH)
X_train_scaled = np.load(BACKGROUND_PATH)
# =========================
# Preprocessing Function
# =========================
def preprocess_input(input_data):
    # Convert dictionary to dataframe
    input_df = pd.DataFrame([input_data])
    # One-hot encoding
    input_df = pd.get_dummies(input_df)
    # Align columns with training data
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )
    # Scale data
    input_scaled = scaler.transform(input_df)
    return input_scaled

# =========================
# Prediction Function
# =========================

def prediction_churn(input_data):

    processed_data = preprocess_input(
        input_data
    )
    prediction_prob = model.predict(
        processed_data
    )
    prediction = (
        prediction_prob > 0.5
    ).astype(int)
    return (
        prediction[0][0],
        prediction_prob[0][0]
    )

