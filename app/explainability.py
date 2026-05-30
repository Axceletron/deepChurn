import shap
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from utils import (
    model,
    X_train_scaled,
    feature_columns
)


@st.cache_resource
def load_explainer():

    background_data = X_train_scaled[:100]

    explainer = shap.Explainer(
        model.predict,
        background_data
    )

    return explainer


explainer = load_explainer()

def generate_shap_plot(sample):

    sample_df = pd.DataFrame(
        sample,
        columns=feature_columns
    )

    shap_values = explainer(sample_df)

    single_explanation = shap_values[0]

    # Debugging
    print("VALUES SHAPE:", np.shape(single_explanation.values))
    print("BASE SHAPE:", np.shape(single_explanation.base_values))
    print("BASE VALUES:", single_explanation.base_values)

    # Normalize shapes
    fixed_explanation = shap.Explanation(
        values=np.squeeze(single_explanation.values),
        base_values=float(np.squeeze(single_explanation.base_values)),
        data=np.squeeze(single_explanation.data),
        feature_names=single_explanation.feature_names
    )

    plt.figure(figsize=(12, 6))

    shap.plots.waterfall(
        fixed_explanation,
        max_display=31,
        show=False
    )

    plt.tight_layout()

    return plt.gcf()