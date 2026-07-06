import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-08-01'
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

LOOKBACK_DAYS = 60
FUTURE_DAYS = 5 #k
TARGET_COL = 'Close'
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

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

for col in FEATURE_COLS:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

data.dropna(subset=FEATURE_COLS, inplace=True)

feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

scaled_features = feature_scaler.fit_transform(data[FEATURE_COLS].values)

scaled_target = target_scaler.fit_transform(data[[TARGET_COL]].values)

train_data_len = len(data[data.index <= TRAIN_END])
train_features = scaled_features[:train_data_len]
train_target = scaled_target[:train_data_len]

x_train, y_train = [], []
for i in range(LOOKBACK_DAYS, len(train_features) - FUTURE_DAYS + 1):
    x_train.append(train_features[i - LOOKBACK_DAYS:i])
    y_train.append(train_target[i:i + FUTURE_DAYS, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

def build_advanced_model(input_shape, output_steps):
    """
    Builds a DL model capable of multivariate input and multistep output.
    """
    print(f"\n[Task C.5] Building model for input {input_shape} and predicting {output_steps} steps...")
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=64, return_sequences=False),
        Dropout(0.2),

        Dense(units=output_steps)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_advanced_model(input_shape=(x_train.shape[1], x_train.shape[2]), output_steps=FUTURE_DAYS)

print("Training the model...")
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

test_features = scaled_features[train_data_len - LOOKBACK_DAYS:]
test_target_actual = data[TARGET_COL].values[train_data_len:]

x_test = []

for i in range(LOOKBACK_DAYS, len(test_features) - FUTURE_DAYS + 1):
    x_test.append(test_features[i - LOOKBACK_DAYS:i])

x_test = np.array(x_test)

predicted_scaled = model.predict(x_test)
predicted_prices = target_scaler.inverse_transform(predicted_scaled)

plt.figure(figsize=(14, 7))
plt.plot(data.index[train_data_len:], test_target_actual, color='black', label=f'Actual {TARGET_COL} Price')

plot_indices = data.index[train_data_len:]
for i in range(0, len(predicted_prices), FUTURE_DAYS):
    pred_window = predicted_prices[1]
    plt.plot(plot_indices[i:i + FUTURE_DAYS], pred_window, color='red', alpha=0.7)

plt.plot([], [], color='red', label=f'Predicted {FUTURE_DAYS}-Day Windows')

plt.title(f"{COMPANY} - Multivariate & Multistep Prediction (Task C.5)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

last_window = np.array([scaled_features[-LOOKBACK_DAYS:]])
future_prediction = model.predict(last_window)
future_prediction = target_scaler.inverse_transform(future_prediction)
print(f"\nPredicted Close prices for the next {FUTURE_DAYS} days:\n{future_prediction[0]}")