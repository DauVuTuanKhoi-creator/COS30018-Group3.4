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

print("\nGenerating LSTM predictions for Meta-Learner...")

lstm_train_preds_scaled = lstm_model.predict(x_train_lstm)
lstm_train_preds = scaler.inverse_transform(lstm_train_preds_scaled).flatten()

test_scaled = scaled_total[len(train_values) - PREDICTION_DAYS:]
x_test_lstm = []
for x in range(PREDICTION_DAYS, len(test_scaled)):
    x_test_lstm.append(test_scaled[x - PREDICTION_DAYS:x, 0])

x_test_lstm = np.array(x_test_lstm)
x_test_lstm = np.reshape(x_test_lstm, (x_test_lstm.shape[0], x_test_lstm.shape[1], 1))

lstm_test_preds_scaled = lstm_model.predict(x_test_lstm)
lstm_test_preds = scaler.inverse_transform(lstm_test_preds_scaled).flatten()

actual_y_train_for_rf = train_values[PREDICTION_DAYS:]
lag_1_train = train_values[PREDICTION_DAYS - 1 : -1]

X_meta_train = np.column_stack((lstm_train_preds, lag_1_train))

lag_1_test = data['Close'].values[len(train_values) - 1 : len(train_values) - 1 + len(test_values)]
X_meta_test = np.column_stack((lstm_test_preds, lag_1_test))

print("\n--- Training Meta Learner: Random Forest ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_meta_train, actual_y_train_for_rf)

ensemble_rf_predictions = rf_model.predict(X_meta_test)

rmse_lstm = np.sqrt(mean_squared_error(test_values, lstm_test_preds))
rmse_rf_ensemble = np.sqrt(mean_squared_error(test_values, ensemble_rf_predictions))

print(f"\n--- EVALUATION ---")
print(f"Base LSTM RMSE: {rmse_lstm:.2f}")
print(f"Meta-Learner Ensemble (LSTM + RF) RMSE: {rmse_rf_ensemble:.2f}")

plt.figure(figsize=(14, 7))
plt.plot(test_data.index, test_values, color='black', label='Actual Price', linewidth=2)
plt.plot(test_data.index, lstm_test_preds, color='blue', alpha=0.5, label='LSTM Base Prediction')
plt.plot(test_data.index, ensemble_rf_predictions, color='red', label='Ensemble (LSTM + Random Forest)', linewidth=2)

plt.title(f"{COMPANY} Stock Price Prediction - Ensemble Approach (Task C.6)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()