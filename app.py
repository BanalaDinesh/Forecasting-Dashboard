
import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Forecasting App", layout="wide")
st.title("📈 Stock Market Forecasting App")

# --- User Inputs ---
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))
model_choice = st.selectbox("Select Forecasting Model", ["Prophet", "ARIMA", "SARIMA"])

if st.button("Generate Forecast"):
    # --- Load Data ---
    st.subheader(f"🔍 Historical Data for {ticker}")
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    st.dataframe(df.tail())

    # --- Plot Historical Prices ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=f"{ticker} Stock Prices", xaxis_title="Date", yaxis_title="Close Price")
    st.plotly_chart(fig)

    # --- Model: Prophet ---
    if model_choice == "Prophet":
        st.subheader("📘 Prophet Forecast")
        prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)

        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

    # --- Model: ARIMA ---
    elif model_choice == "ARIMA":
        st.subheader("📗 ARIMA Forecast")

        close_series = df.set_index('Date')['Close']
        model = ARIMA(close_series, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=90)

        st.write(forecast.tail())

        fig2, ax = plt.subplots()
        close_series.plot(label='Historical', ax=ax)
        forecast.plot(label='Forecast', ax=ax)
        plt.legend()
        st.pyplot(fig2)

    # --- Model: SARIMA ---
    elif model_choice == "SARIMA":
        st.subheader("📕 SARIMA Forecast")

        close_series = df.set_index('Date')['Close']
        model = SARIMAX(close_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=90)

        st.write(forecast.tail())

        fig3, ax = plt.subplots()
        close_series.plot(label='Historical', ax=ax)
        forecast.plot(label='Forecast', ax=ax)
        plt.legend()
        st.pyplot(fig3)
        