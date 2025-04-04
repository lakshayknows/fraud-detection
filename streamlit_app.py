# Required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and preprocessors
dt_model = joblib.load("decision_tree_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
lr_model = joblib.load("logistic_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# UI Layout
st.title("💸 Fraud Detection App")
st.write("Enter the transaction details below:")

# --- User Inputs ---
input_dict = {
    'step': st.number_input("Step", min_value=1),
    "type": st.selectbox("Transaction Type", ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]),
    "amount": st.number_input("Amount", min_value=0.0),
    "oldbalanceOrg": st.number_input("Old Balance Origin", min_value=0.0),
    "newbalanceOrig": st.number_input("New Balance Origin", min_value=0.0),
    "oldbalanceDest": st.number_input("Old Balance Destination", min_value=0.0),
    "newbalanceDest": st.number_input("New Balance Destination", min_value=0.0),
    "nameOrig": st.text_input("Name Origin (e.g. C123456789)", "C123456789"),
    "nameDest": st.text_input("Name Destination (e.g. M987654321)", "M987654321"),
    "isFlaggedFraud": 0,
    "isFraud": 0
}

input_data = pd.DataFrame([input_dict])

# --- Preprocessing Functions ---
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = max(Q1 - 1.5 * IQR, 0)
    return series.clip(lower=lower_limit, upper=upper_limit)

def preprocess_input(df):
    df = df.copy()

    # Outlier Capping
    log_transform_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    for col in log_transform_cols:
        df[col] = cap_outliers(df[col])

    # Log Transformation
    for col in log_transform_cols:
        df[col] = np.log1p(df[col].clip(lower=0))

    # Feature Engineering
    df["balance_change_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_change_dest"] = df["oldbalanceDest"] - df["newbalanceDest"]

    df["nameOrig_num"] = df["nameOrig"].str.extract(r"(\d+)").astype(float)
    df["nameOrig_cat"] = df["nameOrig"].str.extract(r"([A-Za-z]+)")

    df["nameDest_num"] = df["nameDest"].str.extract(r"(\d+)").astype(float)
    df["nameDest_cat"] = df["nameDest"].str.extract(r"([A-Z])")

    # Drop only what’s NOT needed — keep isFraud and isFlaggedFraud
    df.drop(columns=["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "nameOrig", "nameDest"], inplace=True)

    return df

# --- Encode, Scale, and Predict ---
model_choice = st.selectbox("Choose a model", ["Random Forest", "XGBoost", "Logistic Regression"])

if st.button("Predict Fraud"):
    try:
        # Preprocess
        processed_input = preprocess_input(input_data)
        encoded_input = encoder.transform(processed_input)
        scaled_input = scaler.transform(encoded_input)

        # Predict
        if model_choice == "Decision Tree":
            prediction = dt_model.predict(scaled_input)[0]
        elif model_choice == "XGBoost":
            prediction = xgb_model.predict(scaled_input)[0]
        else:
            prediction = lr_model.predict(scaled_input)[0]

        # Show result
        if prediction == 1:
            st.error("⚠️ This transaction is predicted to be FRAUDULENT!")
        else:
            st.success("✅ This transaction is predicted to be GENUINE.")

    except Exception as e:
        st.exception(f"🚨 Something went wrong: {e}")
