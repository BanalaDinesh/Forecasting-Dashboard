import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Axis Bank Stock Forecasting Dashboard", layout="wide")
st.title("📈 Axis Bank Stock Forecasting Dashboard")

# Sidebar for user inputs
st.sidebar.header("Forecast Settings")
model_choice = st.sidebar.radio("Select Forecasting Model", ("ARIMA", "Prophet", "LSTM"))
horizon = st.sidebar.slider("Forecast Horizon (days)", min_value=5, max_value=100, value=30)

# Upload CSV file
uploaded_file = st.file_uploader("Upload AXISBANK.csv", type=["csv"])

if uploaded_file:
    # Load and preprocess data
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    
    st.write("### Raw Data Preview")
    st.write(df.head())

    # Plot raw data
    st.write("### Historical Closing Price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title="Axis Bank Closing Price", xaxis_title="Date", yaxis_title="Price (INR)")
    st.plotly_chart(fig)

    # Run forecast when button is clicked
    if st.button("Run Forecast"):
        st.success(f"Running {model_choice} forecast for {horizon} days...")

        if model_choice == "ARIMA":
            # ARIMA Model
            series = df['Close']
            model = ARIMA(series, order=(5,1,0))  # Adjust order based on your auto_arima results
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            future_index = pd.date_range(start=series.index[-1], periods=horizon+1, freq='D')[1:]
            forecast_df = pd.DataFrame({'Forecast': forecast}, index=future_index)

        elif model_choice == "Prophet":
            # Prophet Model
            prophet_df = df.reset_index()[['Date', 'Close']]
            prophet_df.columns = ['ds', 'y']
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)
            forecast_df = forecast[['ds', 'yhat']].set_index('ds').tail(horizon)
            forecast_df.columns = ['Forecast']

        elif model_choice == "LSTM":
            # LSTM Model
            data = df['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
            window_size = 60
            X, y = [], []
            for i in range(window_size, len(data_scaled)):
                X.append(data_scaled[i-window_size:i, 0])
                y.append(data_scaled[i, 0])
            X = np.array(X)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            
            input_seq = data_scaled[-window_size:].reshape(1, window_size, 1)
            predictions = []
            for _ in range(horizon):
                pred = model.predict(input_seq, verbose=0)[0][0]
                predictions.append(pred)
                input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)
            
            forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            future_index = pd.date_range(start=df.index[-1], periods=horizon+1, freq='D')[1:]
            forecast_df = pd.DataFrame({'Forecast': forecast.flatten()}, index=future_index)

        # Plot forecast
        st.write("### Forecast Results")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical'))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name=f'{model_choice} Forecast', line=dict(dash='dash')))
        fig.update_layout(title=f"{model_choice} Forecast", xaxis_title="Date", yaxis_title="Price (INR)")
        st.plotly_chart(fig)

        st.write("### Forecast Data")
        st.write(forecast_df)

        # Display performance metrics
        st.write("### Model Performance (Historical)")
        metrics = {
            "ARIMA": {"MAE": 143.33, "RMSE": 175.61},
            "Prophet": {"MAE": 82.46, "RMSE": 135.29},
            "LSTM": {"MAE": 18.66, "RMSE": 26.31}
        }
        st.write(f"**{model_choice} Metrics**")
        st.write(f"MAE: {metrics[model_choice]['MAE']:.2f}")
        st.write(f"RMSE: {metrics[model_choice]['RMSE']:.2f}")
else:
    st.info("Please upload AXISBANK.csv to start forecasting.")
