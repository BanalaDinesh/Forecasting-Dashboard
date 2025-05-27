
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .title {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        color: #117A65;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">📊 Interactive Stock Market Forecast Dashboard</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your stock CSV file with 'Date' and 'Close' columns", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'Date' not in data.columns or 'Close' not in data.columns:
        st.error("CSV must contain 'Date' and 'Close' columns.")
    else:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        st.markdown('<div class="section-header">Historical Stock Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(data.tail(10))

        # Historical Plot with Plotly for interactivity
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', line=dict(color='#007ACC')))
        fig_hist.update_layout(title='Closing Prices Over Time', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_hist, use_container_width=True)

        # Forecasting horizon
        forecast_days = 30

        # -------- ARIMA --------
        st.markdown('<div class="section-header">ARIMA Forecast</div>', unsafe_allow_html=True)
        model_arima = ARIMA(data['Close'], order=(5, 1, 0))
        result_arima = model_arima.fit()
        forecast_arima = result_arima.forecast(steps=forecast_days)

        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical'))
        future_dates_arima = pd.date_range(start=data.index[-1], periods=forecast_days + 1, closed='right')
        fig_arima.add_trace(go.Scatter(x=future_dates_arima, y=forecast_arima, mode='lines', name='ARIMA Forecast', line=dict(dash='dot')))
        fig_arima.update_layout(title='ARIMA Forecast vs Historical', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_arima, use_container_width=True)

        rmse_arima = np.sqrt(mean_squared_error(data['Close'][-forecast_days:], forecast_arima[:forecast_days]))

        # -------- SARIMA --------
        st.markdown('<div class="section-header">SARIMA Forecast</div>', unsafe_allow_html=True)
        model_sarima = SARIMAX(data['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
        result_sarima = model_sarima.fit(disp=False)
        forecast_sarima = result_sarima.forecast(steps=forecast_days)

        fig_sarima = go.Figure()
        fig_sarima.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical'))
        future_dates_sarima = pd.date_range(start=data.index[-1], periods=forecast_days + 1, closed='right')
        fig_sarima.add_trace(go.Scatter(x=future_dates_sarima, y=forecast_sarima, mode='lines', name='SARIMA Forecast', line=dict(dash='dash')))
        fig_sarima.update_layout(title='SARIMA Forecast vs Historical', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_sarima, use_container_width=True)

        rmse_sarima = np.sqrt(mean_squared_error(data['Close'][-forecast_days:], forecast_sarima[:forecast_days]))

        # -------- Prophet --------
        st.markdown('<div class="section-header">Prophet Forecast</div>', unsafe_allow_html=True)
        prophet_df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        model_prophet = Prophet()
        model_prophet.fit(prophet_df)

        future_prophet = model_prophet.make_future_dataframe(periods=forecast_days)
        forecast_prophet = model_prophet.predict(future_prophet)

        fig_prophet = go.Figure()
        fig_prophet.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical'))
        fig_prophet.add_trace(go.Scatter(x=forecast_prophet['ds'].tail(forecast_days), y=forecast_prophet['yhat'].tail(forecast_days),
                                        mode='lines', name='Prophet Forecast', line=dict(color='firebrick')))
        fig_prophet.update_layout(title='Prophet Forecast vs Historical', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_prophet, use_container_width=True)

        rmse_prophet = np.sqrt(mean_squared_error(data['Close'][-forecast_days:], forecast_prophet['yhat'].tail(forecast_days)))

        # -------- LSTM --------
        st.markdown('<div class="section-header">LSTM Forecast</div>', unsafe_allow_html=True)

        # Data scaling
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data[['Close']])

        # Prepare training data (80%) and test data (20%)
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - 60:]  # Include some past 60 steps for sequences

        def create_sequences(dataset, seq_length=60):
            X, y = [], []
            for i in range(seq_length, len(dataset)):
                X.append(dataset[i - seq_length:i, 0])
                y.append(dataset[i, 0])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_data)
        X_test, y_test = create_sequences(test_data)

        # Reshape input to be [samples, time_steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Build LSTM model
        model_lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        with st.spinner("Training LSTM model (this may take a moment)..."):
            model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Predict
        predictions_scaled = model_lstm.predict(X_test)
        predictions = scaler.inverse_transform(predictions_scaled)

        # Prepare date index for forecast
        forecast_index = data.index[train_size + 60:]

        # Plot LSTM results
        fig_lstm = go.Figure()
        fig_lstm.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical'))
        fig_lstm.add_trace(go.Scatter(x=forecast_index, y=predictions.flatten(), mode='lines', name='LSTM Prediction', line=dict(color='green')))
        fig_lstm.update_layout(title='LSTM Forecast vs Historical', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_lstm, use_container_width=True)

        # RMSE for LSTM (last forecast_days or test length, whichever is smaller)
        test_len = min(forecast_days, len(y_test))
        rmse_lstm = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1,1))[:test_len], predictions[:test_len]))

        # -------- Model Accuracy Summary --------
        st.markdown('<div class="section-header">Model Accuracy (RMSE)</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <ul>
                <li><b>ARIMA:</b> {rmse_arima:.4f}</li>
                <li><b>SARIMA:</b> {rmse_sarima:.4f}</li>
                <li><b>Prophet:</b> {rmse_prophet:.4f}</li>
                <li><b>LSTM:</b> {rmse_lstm:.4f}</li>
            </ul>
            """, unsafe_allow_html=True)

        st.success("Forecasts generated successfully!")
else:
    st.info("Please upload a CSV file to start forecasting.")
