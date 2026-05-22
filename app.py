import pandas as pd

uploaded_file = st.file_uploader("Upload transaction CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    prob = model.predict_proba(df)[:,1]
    pred = (prob >= threshold).astype(int)

    df["Fraud Probability"] = prob
    df["Prediction"] = pred

    st.write(df.head())