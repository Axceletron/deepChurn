# Customer Churn Prediction using ANN

## Project Overview

This project focuses on predicting customer churn using an Artificial Neural Network (ANN).

The goal is to build an end-to-end Deep Learning project covering:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- ANN model training
- Model evaluation
- Deployment

Current Progress:
- ✅ Project setup completed
- ✅ Dataset imported
- ✅ Exploratory Data Analysis (EDA) completed
- ⏳ Feature Engineering in progress
- ⏳ ANN model training pending
- ⏳ Deployment pending

---

# Dataset

Dataset Used:
IBM Telco Customer Churn Dataset

The dataset contains customer-related information such as:

- Gender
- Senior citizen status
- Tenure
- Internet service
- Contract type
- Monthly charges
- Total charges
- Churn status

Target Variable:
```text
Churn
```

---

# EDA Steps Performed

## 1. Dataset Inspection

Performed:

- Shape analysis
- Column inspection
- Datatype checking
- Statistical summaries

---

## 2. Missing Value Handling

Identified issues in:

```text
TotalCharges
```

Converted invalid values to numeric and removed null rows.

---

## 3. Churn Distribution Analysis

Analyzed class imbalance between:

- Churn = Yes
- Churn = No

### Observation

- Dataset is imbalanced.

---

## 4. Feature Analysis

Analyzed important features including:

- tenure
- MonthlyCharges
- Contract type

### Visualizations Created

- Count plots
- Histograms
- Boxplots

---

## 5. Correlation Analysis

Generated correlation heatmap after temporary encoding of categorical features to identify relationships between variables.

---

# Key Insights from EDA

- Customers with month-to-month contracts show higher churn.
- Customers with lower tenure are more likely to churn.
- Higher monthly charges are associated with increased churn.
- Long-term contracts reduce churn probability.
- Dataset contains class imbalance.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Next Steps

Upcoming work:

- Data Preprocessing
- ANN Model Building
- Model Evaluation
- Streamlit Deployment

---

# Author
S Raj Shekhar
