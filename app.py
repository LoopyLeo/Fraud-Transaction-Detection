import streamlit as st
import pickle
import pandas as pd
import numpy as np

# LOAD MODEL + THRESHOLD

model = pickle.load(open("model.pkl", "rb"))
threshold = pickle.load(open("threshold.pkl", "rb"))

st.title("💳 Fraud Transaction Detection")


# MODE SELECTION

mode = st.radio("Choose Input Mode:", ["Upload Dataset", "Manual Input"])


#  MODE 1: DATASET UPLOAD

if mode == "Upload Dataset":

    uploaded_file = st.file_uploader("Upload File (.csv or .pkl)", type=["csv", "pkl"])

    if uploaded_file is not None:

        # READ FILE
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_pickle(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        st.write("### Raw Data Preview")
        st.write(df.head())

        # SELECT FEATURES
        expected_cols = ["TX_AMOUNT", "TX_TIME_SECONDS", "TX_TIME_DAYS"]

        try:
            df = df[expected_cols]
        except Exception as e:
            st.error(f"Column Error: {e}")
            st.stop()

        # CLEAN
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.fillna(0)

        # PREDICT
        prob = model.predict_proba(df)[:, 1]
        pred = (prob >= threshold).astype(int)

        df["Fraud Probability"] = prob
        df["Prediction"] = pred

        st.write("### Predictions")
        st.write(df.head())

        st.warning(f"🚨 Fraud Transactions: {int(np.sum(pred))} / {len(pred)}")


#  MODE 2: MANUAL INPUT

else:

    st.write("Enter transaction details:")

    tx_amount = st.number_input("Transaction Amount", min_value=0.0)
    tx_time_seconds = st.number_input("Time (Seconds)", min_value=0)
    tx_time_days = st.number_input("Time (Days)", min_value=0)

    if st.button("Predict Fraud"):

        input_data = pd.DataFrame([{
            "TX_AMOUNT": tx_amount,
            "TX_TIME_SECONDS": tx_time_seconds,
            "TX_TIME_DAYS": tx_time_days
        }])

        # CLEAN
        input_data = input_data.apply(pd.to_numeric, errors="coerce")
        input_data = input_data.fillna(0)

        # PREDICT
        prob = model.predict_proba(input_data)[0][1]
        pred = 1 if prob >= threshold else 0

        st.write(f"Fraud Probability: {prob:.4f}")

        if pred == 1:
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Legit Transaction")