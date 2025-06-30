import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from datetime import datetime as dt

plt.style.use("fivethirtyeight")

# Load model
model = joblib.load("model/linear_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(page_title="📈 Stock Predictor (Linear Model)", layout="wide")
st.title("🔮 Stock Price Prediction using Linear Regression")

st.markdown("Enter stock symbol (e.g., `POWERGRID.NS`, `TCS.NS`) to get prediction + technical analysis.")

stock = st.text_input("📌 Enter Stock Symbol:", "POWERGRID.NS")

if st.button("Predict"):
    try:
        start = dt(2010, 1, 1)
        end = dt(2024, 11, 1)
        df = yf.download(stock, start=start, end=end)

        if df.empty:
            st.error("❌ No data found. Check stock symbol.")
            st.stop()

        df.reset_index(inplace=True)
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['MA100'] = df['Close'].rolling(window=100).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df.dropna(inplace=True)

        # Show data
        st.subheader("📊 Data Preview")
        st.dataframe(df.tail())

        # Predict next closing price
        features = ['Open', 'High', 'Low', 'Volume', 'MA100', 'MA200', 'EMA100', 'EMA200']
        latest = df[features].iloc[-1:]
        latest_scaled = scaler.transform(latest)
        pred_price = model.predict(latest_scaled)[0]

        st.success(f"📈 Predicted Next Day Closing Price for `{stock}`: ₹{pred_price:.2f}")

        # Plot 1: EMA 20 & 50
        st.subheader("📉 Plot 1: Closing Price with EMA 20 & 50")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df['Date'], df['Close'], label='Close', color='skyblue')
        ax1.plot(df['Date'], df['EMA20'], label='EMA 20', color='orange')
        ax1.plot(df['Date'], df['EMA50'], label='EMA 50', color='green')
        ax1.legend()
        ax1.set_title("Close Price + EMA 20/50")
        st.pyplot(fig1)

        # Plot 2: MA 100 & 200
        st.subheader("📉 Plot 2: Closing Price with MA 100 & 200")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df['Date'], df['Close'], label='Close', color='skyblue')
        ax2.plot(df['Date'], df['MA100'], label='MA 100', color='purple')
        ax2.plot(df['Date'], df['MA200'], label='MA 200', color='brown')
        ax2.legend()
        ax2.set_title("Close Price + MA 100/200")
        st.pyplot(fig2)

        # 🔮 Plot 3: Prediction vs Actual Closing Price
        st.subheader("🔮 Plot 3: Prediction vs Actual Closing Price")

        fig3, ax3 = plt.subplots(figsize=(12, 6))

        # Plot actual close price
        ax3.plot(df['Date'], df['Close'], label='Actual Close Price', color='green')

        # Get scalar values
        last_date = df['Date'].iloc[-1]
        last_close = float(df['Close'].iloc[-1])
        next_day = last_date + pd.Timedelta(days=1)
        predicted_price = float(pred_price)  # Ensure it's not an array

        # Plot line from last point to predicted point
        ax3.plot([last_date, next_day], [last_close, predicted_price],label='Predicted Price (Next Day)', color='red', linestyle='--', marker='X')
        

        ax3.set_title("Prediction vs Actual Closing Price")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Price")
        ax3.legend()
        st.pyplot(fig3)


        # Save and offer CSV download
        csv_path = f"static/{stock}_data.csv"
        os.makedirs("static", exist_ok=True)
        df.to_csv(csv_path, index=False)

        with open(csv_path, 'rb') as f:
            st.download_button("📥 Download Dataset", f, file_name=f"{stock}_data.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
