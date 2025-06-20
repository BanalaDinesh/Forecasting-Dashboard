

#Installing Required libraries


pip install pandas numpy matplotlib seaborn plotly scikit-learn statsmodels prophet keras tensorflow

"""#Preprocessing"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn.set() to apply seaborn's styles
sns.set()


# %matplotlib inline

"""#Installing statsmodels"""

pip install statsmodels

"""#Seasonal Decomposition / EDA"""

from statsmodels.tsa.seasonal import seasonal_decompose


result = seasonal_decompose(df['Close'], model='multiplicative', period=30)
result.plot()
plt.tight_layout()
plt.show()

"""#Time series forcasting using Arima"""

pip install pmdarima

from statsmodels.tsa.stattools import adfuller

# Perform ADF test on 'Close' column
adf_result = adfuller(df['Close'])
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
df['Close_diff'] = df['Close'].diff().dropna()
adf_result_diff = adfuller(df['Close_diff'].dropna())
print(f"ADF Statistic (after differencing): {adf_result_diff[0]}")
print(f"p-value (after differencing): {adf_result_diff[1]}")

!pip uninstall -y numpy pmdarima
!pip install numpy --upgrade
!pip install pmdarima
!pip install numpy==1.23.5
!pip install pmdarima

from google.colab import files
uploaded = files.upload()
import pandas as pd

df = pd.read_csv("AXISBANK.csv")
df.head()
from pmdarima import auto_arima

stepwise_model = auto_arima(df['Close'],
                            start_p=1, start_q=1,
                            max_p=3, max_q=3,
                            seasonal=False,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

print(stepwise_model.summary())

# Split data: 80% train, 20% test
train_size = int(len(df) * 0.8)
train, test = df['Close'][:train_size], df['Close'][train_size:]

# Fit ARIMA model
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train, order=stepwise_model.order)
model_fit = model.fit()


forecast = model_fit.forecast(steps=len(test))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.legend()
plt.title("ARIMA Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

"""#Time series forecasting using sarima"""

from google.colab import files
uploaded = files.upload()

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
df = pd.read_csv('AXISBANK.csv')

# Parse dates
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
else:
    raise KeyError("The CSV must contain a 'Date' column.")

# Ensure daily frequency and handle missing values
df = df.asfreq('D') 
df['Close'] = df['Close'].interpolate(method='linear')

# Plot the original data
df['Close'].plot(figsize=(12, 5), title='Close Price Time Series')

# Building and fitting SARIMA model
model = SARIMAX(df['Close'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),  
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)

# Forecast the next 30 days
n_steps = 30
forecast = results.get_forecast(steps=n_steps)
conf_int = forecast.conf_int()

# Plot the forecast
plt.figure(figsize=(12, 5))
plt.plot(df['Close'], label='Historical')
plt.plot(forecast.predicted_mean, label='SARIMA Forecast')
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('SARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


train_size = int(len(df) * 0.9)
train = df['Close'][:train_size]
test = df['Close'][train_size:]

model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)


preds = results.get_forecast(steps=len(test))
sarima_pred = preds.predicted_mean
y_true = test  # Actual test values


def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

evaluate_model("SARIMA", y_true, sarima_pred)

"""##Time series forcasting using Prophet"""

pip install prophet


prophet_df = df.reset_index()[['Date', 'Close']]
prophet_df.columns = ['ds', 'y']

prophet_df.head()

from prophet import Prophet


model = Prophet()

# Fit model
model.fit(prophet_df)

forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("Prophet Forecast for Stock Price")
plt.show()

model.plot_components(forecast)
plt.show()

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from prophet import Prophet

prophet_df = df.reset_index()[['Date', 'Close']]
prophet_df.columns = ['ds', 'y']

prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])


# Fit model
model = Prophet()
model.fit(prophet_df)

# Predict (historical + future)
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)


df_eval = prophet_df.merge(forecast[['ds', 'yhat']], on='ds')
mae = mean_absolute_error(df_eval['y'], df_eval['yhat'])
rmse = np.sqrt(mean_squared_error(df_eval['y'], df_eval['yhat']))

print(f"Prophet MAE: {mae:.2f}")
print(f"Prophet RMSE: {rmse:.2f}")

"""##Time series forcasting using LSTM"""

pip install tensorflow keras scikit-learn

pip uninstall -y numpy jax

pip install tensorflow

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()
import pandas as pd

df = pd.read_csv('AXISBANK.csv') 
df.head()
from sklearn.preprocessing import MinMaxScaler
# Use only 'Close' column
data = df[['Close']].values

# Normalize between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create training data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:] 

# Create sequences
def create_dataset(data, time_step=60):
    x, y = [], []
    for i in range(time_step, len(data)):
        x.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Reshape for LSTM: [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)
# Predict
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert scaling
train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform([y_test])
# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[0], label='Actual')
plt.plot(test_predict, label='Predicted')
plt.title("LSTM Stock Price Forecast")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(y_test_actual[0], test_predict[:, 0]))
mae = mean_absolute_error(y_test_actual[0], test_predict[:, 0])

print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")

"""#Mae and rmse values  of arima,prophet and LSTM"""

arima_mae = 143.33064971751412
arima_rmse = 175.60910936347028

prophet_mae = 82.46
prophet_rmse =  135.29

lstm_mae = 18.663658920834067
lstm_rmse = 26.307071281734014

sarima_mae = 159.3175
sarima_rmse = 211.7818

"""#Comaparision of Arima,Prophet and LSTM"""

import matplotlib.pyplot as plt

# List of models
models = ['ARIMA', 'Prophet', 'LSTM', 'SARIMA']

# Corresponding MAE and RMSE values (ensure these variables are defined)
mae_scores = [arima_mae, prophet_mae, lstm_mae, sarima_mae]
rmse_scores = [arima_rmse, prophet_rmse, lstm_rmse, sarima_rmse]

# Plot MAE Comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(models, mae_scores, color=['blue', 'green', 'orange', 'purple'])
plt.title('Model MAE Comparison')
plt.ylabel('MAE (lower is better)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Plot RMSE Comparison
plt.subplot(1, 2, 2)
plt.bar(models, rmse_scores, color=['blue', 'green', 'orange', 'purple'])
plt.title('Model RMSE Comparison')
plt.ylabel('RMSE (lower is better)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

"""#Stream lit Dashboard

"""

pip install streamlit yfinance pandas prophet matplotlib plotly

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
import os
from plotly.subplots import make_subplots

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

st.markdown('<div class="title">📊 Stock Market Forecast Dashboard</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your stock CSV file with 'Date' and 'Close' columns", type=["csv"])

if uploaded_file is not None:
    stock_name = os.path.splitext(uploaded_file.name)[0].upper()
    data = pd.read_csv(uploaded_file)
    if 'Date' not in data.columns or 'Close' not in data.columns:
        st.error("CSV must contain 'Date' and 'Close' columns.")
    else:
        data['Date'] = pd.to_datetime(data['Date'])
        min_date = data['Date'].min().date()
        max_date = data['Date'].max().date()

        start_date, end_date = st.date_input(
            "Select date range for analysis:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        mask = (data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))
        data = data.loc[mask]
        data.set_index('Date', inplace=True)

        st.markdown('<div class="section-header">Stock Data Summary</div>', unsafe_allow_html=True)
        mean_price = data['Close'].mean()
        st.markdown(f"""
        - 🏷️ **Stock Name:** {stock_name}
        - 📅 **Date Range:** {data.index.min().date()} to {data.index.max().date()}
        - 📈 **Number of Records:** {len(data)}
        - 💰 **Last Close Price:** <span title="Mean over selected range: {mean_price:.2f}">{data['Close'].iloc[-1]:.2f}</span>
        """, unsafe_allow_html=True)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=data.index, y=data['Close'], mode='lines',
            line=dict(color='#007ACC'),
            hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ))
        fig_hist.update_layout(title='Closing Prices Over Time', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_hist, use_container_width=True)

        forecast_days = 30

        # ARIMA
        st.markdown('<div class="section-header">ARIMA Forecast</div>', unsafe_allow_html=True)
        model_arima = ARIMA(data['Close'], order=(5, 1, 0))
        result_arima = model_arima.fit()
        forecast_arima = result_arima.forecast(steps=forecast_days)
        future_dates_arima = pd.date_range(start=data.index[-1], periods=forecast_days + 1)

        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(
            x=data.index, y=data['Close'], mode='lines', name='Historical',
            hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ))
        fig_arima.add_trace(go.Scatter(
            x=future_dates_arima, y=forecast_arima, mode='lines', name='ARIMA Forecast',
            line=dict(dash='dot'),
            hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ))

        fig_arima.update_layout(title='ARIMA Forecast vs Historical', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_arima, use_container_width=True)

        rmse_arima = np.sqrt(mean_squared_error(data['Close'][-forecast_days:], forecast_arima[:forecast_days]))

        # SARIMA
        st.markdown('<div class="section-header">SARIMA Forecast</div>', unsafe_allow_html=True)
        model_sarima = SARIMAX(data['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
        result_sarima = model_sarima.fit(disp=False)
        forecast_sarima = result_sarima.forecast(steps=forecast_days)
        future_dates_sarima = pd.date_range(start=data.index[-1], periods=forecast_days + 1)

        fig_sarima = go.Figure()
        fig_sarima.add_trace(go.Scatter(
            x=data.index, y=data['Close'], mode='lines', name='Historical',
            hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ))
        fig_sarima.add_trace(go.Scatter(
            x=future_dates_sarima, y=forecast_sarima, mode='lines', name='SARIMA Forecast',
            line=dict(dash='dash'),
            hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ))
        fig_sarima.update_layout(title='SARIMA Forecast vs Historical', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_sarima, use_container_width=True)

        rmse_sarima = np.sqrt(mean_squared_error(data['Close'][-forecast_days:], forecast_sarima[:forecast_days]))

        # Prophet
        st.markdown('<div class="section-header">Prophet Forecast</div>', unsafe_allow_html=True)
        prophet_df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        model_prophet = Prophet()
        model_prophet.fit(prophet_df)
        future_prophet = model_prophet.make_future_dataframe(periods=forecast_days)
        forecast_prophet = model_prophet.predict(future_prophet)

        fig_prophet = go.Figure()
        fig_prophet.add_trace(go.Scatter(
            x=data.index, y=data['Close'], mode='lines', name='Historical',
            hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ))
        fig_prophet.add_trace(go.Scatter(
            x=forecast_prophet['ds'].tail(forecast_days), y=forecast_prophet['yhat'].tail(forecast_days),
            mode='lines', name='Prophet Forecast', line=dict(color='firebrick'),
            hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ))
        fig_prophet.update_layout(title='Prophet Forecast vs Historical', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_prophet, use_container_width=True)

        rmse_prophet = np.sqrt(mean_squared_error(data['Close'][-forecast_days:], forecast_prophet['yhat'].tail(forecast_days)))

        # LSTM
        st.markdown('<div class="section-header">LSTM Forecast</div>', unsafe_allow_html=True)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data[['Close']])
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - 60:]

        def create_sequences(dataset, seq_length=60):
            X, y = [], []
            for i in range(seq_length, len(dataset)):
                X.append(dataset[i - seq_length:i, 0])
                y.append(dataset[i, 0])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_data)
        X_test, y_test = create_sequences(test_data)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model_lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')

        with st.spinner("Training LSTM model (this may take a moment)..."):
            model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        predictions_scaled = model_lstm.predict(X_test)
        predictions = scaler.inverse_transform(predictions_scaled)
        forecast_index = data.index[train_size + 60:]

        fig_lstm = go.Figure()
        fig_lstm.add_trace(go.Scatter(
            x=data.index, y=data['Close'], mode='lines', name='Historical',
            hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ))
        fig_lstm.add_trace(go.Scatter(
            x=forecast_index, y=predictions.flatten(), mode='lines', name='LSTM Prediction',
            line=dict(color='green'),
            hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ))
        fig_lstm.update_layout(title='LSTM Forecast vs Historical', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_lstm, use_container_width=True)

        test_len = min(forecast_days, len(y_test))
        rmse_lstm = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1,1))[:test_len], predictions[:test_len]))


        # RMSE Comparison
        st.markdown('<div class="section-header">📊 Model Accuracy (RMSE)</div>', unsafe_allow_html=True)

        rmse_data = {
            'Model': ['ARIMA', 'SARIMA', 'Prophet', 'LSTM'],
            'RMSE': [rmse_arima, rmse_sarima, rmse_prophet, rmse_lstm]
        }

        df_rmse = pd.DataFrame(rmse_data)

        fig_rmse = go.Figure(data=[
            go.Bar(x=df_rmse['Model'], y=df_rmse['RMSE'], marker_color=['#1f77b4', '#2ca02c', '#d62728', '#9467bd'])
        ])
        fig_rmse.update_layout(title="Model RMSE Comparison", xaxis_title="Model", yaxis_title="RMSE", height=400)
        st.plotly_chart(fig_rmse, use_container_width=True)

        st.success("All set! Dive into the trends and take action.")
else:
    st.info("Please upload a CSV file to start forecasting.")

with open("app.py", "w") as f:
    f.write(code)

from google.colab import files
files.download("app.py")
