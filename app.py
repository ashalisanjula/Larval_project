import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Larval Population Monitoring Dashboard", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("population_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model()

st.title("🐛 Larval Population Monitoring Dashboard")
st.write("Upload a CSV or Excel file to analyze larval population and forecast the next 7 days.")

uploaded_file = st.file_uploader("Upload population dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        df.columns = df.columns.str.strip()
        target_column = "Total_Laval_count"

        if target_column not in df.columns:
            st.error(f"Target column '{target_column}' not found in dataset.")
            st.stop()

        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in dataset: {missing_cols}")
            st.stop()

        st.subheader("Dataset Preview")
        st.dataframe(df.tail())

        X = df[feature_columns]
        y = df[target_column]
        X_scaled = scaler.transform(X)

        y_pred_full = model.predict(X_scaled)

        r2 = r2_score(y, y_pred_full)
        mae = mean_absolute_error(y, y_pred_full)

        col1, col2 = st.columns(2)
        col1.metric("R² Score", round(r2, 3))
        col2.metric("MAE", round(mae, 2))

        last_rows = df.iloc[-7:].copy()
        future_preds = []

        for i in range(7):
            X_last = last_rows[feature_columns]
            X_last_scaled = scaler.transform(X_last)
            pred = model.predict([X_last_scaled[-1]])[0]
            future_preds.append(float(pred))

            new_row = last_rows.iloc[-1].copy()
            new_row[target_column] = pred
            last_rows = pd.concat([last_rows, pd.DataFrame([new_row])], ignore_index=True)

        st.subheader("📅 7-Day Population Forecast")
        forecast_df = pd.DataFrame({
            "Day Ahead": [f"Day +{i}" for i in range(1, 8)],
            "Predicted Population": np.round(future_preds, 2)
        })
        st.dataframe(forecast_df)

        st.subheader("📈 Population Trend")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df[target_column], label="Historical")
        ax.plot(range(len(df), len(df) + 7), future_preds, label="Forecast")
        ax.set_xlabel("Time")
        ax.set_ylabel("Total Larval Count")
        ax.set_title("Population Trend")
        ax.legend()
        st.pyplot(fig)

        if future_preds[0] > df[target_column].mean() * 1.5:
            st.error("⚠️ High Outbreak Risk Detected!")
        else:
            st.success("✅ Population within normal range.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload your dataset to start monitoring.")