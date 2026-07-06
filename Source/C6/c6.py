import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore') #Ignore ARIMA convergence warnings

COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-08-01'
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'
PREDICTION_DAYS = 60

DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
file_path = os.path.join(DATA_DIR, f"{COMPANY}_data.csv")

if os.path.exists(file_path):
    print(f"Loading data from local file: {file_path}")
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
else:
    print(f"Downloading data from Yahoo Finance...")
    data = yf.download(COMPANY, start=TRAIN_START, end=TEST_END)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    data.to_csv(file_path)

data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data.dropna(subset=['Close'], inplace=True)

train_data = data[data.index <= TRAIN_END]
test_data = data[data.index > TRAIN_END]

train_values = train_data['Close'].values
test_values = test_data['Close'].values

print("\n--- Training LSTM Model ---")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_total = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

train_scaled = scaled_total[:len(train_values)]
x_train_lstm, y_train_lstm = [], []

for x in range(PREDICTION_DAYS, len(train_scaled)):
    x_train_lstm.append(train_scaled[x - PREDICTION_DAYS:x, 0])
    y_train_lstm.append(train_scaled[x, 0])

x_train_lstm, y_train_lstm = np.array(x_train_lstm), np.array(y_train_lstm)
x_train_lstm = np.reshape(x_train_lstm, (x_train_lstm.shape[0], x_train_lstm.shape[1], 1))

lstm_model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train_lstm.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(x_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=1)

test_scaled = scaled_total[len(train_values) - PREDICTION_DAYS:]
x_test_lstm = []
for x in range(PREDICTION_DAYS, len(test_scaled)):
    x_test_lstm.append(test_scaled[x - PREDICTION_DAYS:x, 0])

x_test_lstm = np.array(x_test_lstm)
x_test_lstm = np.reshape(x_test_lstm, (x_test_lstm.shape[0], x_test_lstm.shape[1], 1))

lstm_predictions = lstm_model.predict(x_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions).flatten()

print("\n--- Training ARIMA Model ---")
history = list(train_values)
arima_predictions = []

print("Forecasting with ARIMA (Walk-forward)... This may take a moment.")
for t in range(len(test_values)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    arima_predictions.append(yhat)
    history.append(test_values[t])

arima_predictions = np.array(arima_predictions)

print("\n--- Generating Ensemble Predictions ---")

WEIGHT_LSTM = 0.6
WEIGHT_ARIMA = 0.4

ensemble_predictions = (lstm_predictions * WEIGHT_LSTM) + (arima_predictions * WEIGHT_ARIMA)

rmse_lstm = np.sqrt(mean_squared_error(test_values, lstm_predictions))
rmse_arima = np.sqrt(mean_squared_error(test_values, arima_predictions))
rmse_ensemble = np.sqrt(mean_squared_error(test_values, ensemble_predictions))

print(f"LSTM RMSE: {rmse_lstm:.2f}")
print(f"ARIMA RMSE: {rmse_arima:.2f}")
print(f"ENSEMBLE RMSE: {rmse_ensemble:.2f}")

plt.figure(figsize=(14, 7))
plt.plot(test_data.index, test_values, color='black', label='Actual Price', linewidth=2)
plt.plot(test_data.index, lstm_predictions, color='blue', alpha=0.5, label='LSTM Prediction')
plt.plot(test_data.index, arima_predictions, color='orange', alpha=0.5, label='ARIMA Prediction')
plt.plot(test_data.index, ensemble_predictions, color='red', label='Ensemble (LSTM + ARIMA)', linewidth=2)

plt.title(f"{COMPANY} Stock Price Prediction - Ensemble Approach (Task C.6)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
