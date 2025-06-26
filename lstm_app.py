# lstm_app.py - Final Streamlit App for TCS Stock Prediction using LSTM

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib

# --- Page Setup ---
st.set_page_config(page_title="TCS Stock Predictor", layout="centered")
st.title("📈 TCS.NS Stock Price Prediction with LSTM")

# --- Load Model and Scaler ---
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model.keras")

@st.cache_resource
def load_scaler():
    scaler = joblib.load("scaler.joblib")
    return scaler

# --- Load and Preprocess Data ---
def load_data():
    symbol = "TCS.NS"
    data = yf.download(symbol, start="2010-01-01")
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return data

def create_dataset(X, y_col_index, window_size=120):
    x, y = [], []
    for i in range(window_size, len(X)):
        x.append(X[i - window_size:i])
        y.append(X[i, y_col_index])
    return np.array(x), np.array(y)

# --- Prediction Function ---
def predict(model, data, scaler):
    scaled_data = scaler.transform(data)
    x_test, y_test = create_dataset(scaled_data, y_col_index=3, window_size=120)
    predictions = model.predict(x_test)

    def inverse_transform(scaled, template):
        temp = np.zeros((len(scaled), template.shape[1]))
        temp[:, 3] = scaled.reshape(-1)
        return scaler.inverse_transform(temp)[:, 3]

    predicted_prices = inverse_transform(predictions, data.values)
    actual_prices = inverse_transform(y_test.reshape(-1, 1), data.values)

    return actual_prices, predicted_prices

# --- Run App ---
try:
    model = load_lstm_model()
    scaler = load_scaler()
    data = load_data()

    actual, predicted = predict(model, data, scaler)

    st.subheader("📉 Actual vs Predicted Closing Prices")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual, label='Actual', color='blue')
    ax.plot(predicted, label='Predicted', color='red')
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"⚠️ An error occurred: {e}")
    st.stop()

st.success("✅ Prediction completed successfully!")
