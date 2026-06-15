import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import tensorflow as tf
import yfinance as yf
import os 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Create an archive folder if you don't already have one
COMPANY = 'CBA.AX'

#Date train
TRAIN_START = '2023-01-01'
TRAIN_END = '2024-08-01'

# Date test
TEST_START = '2024-08-02'
TEST_END = '2025-07-02'

# Folder Name
DATA_DIR = 'data'
MODEL_PATH = 'stock_model.h5'
# C.2: Changed the number of predicted days into 30
PREDICTION_DAYS = 30

# Create folder if not exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# LOAD AND SAVE DATA if the file have been implemented previously
file_path = os.path.join(DATA_DIR, f"{COMPANY}_data.csv")

if os.path.exists(file_path):
    print(f"Loading data from local file: {file_path}")
    data = pd.read_csv(file_path, skiprows=[1, 2], index_col=0, parse_dates=True)
else:
    print(f"Loading data from Yahoo Finance...")
    data = yf.download(COMPANY, start=TRAIN_START, end=TEST_END)
    data.to_csv(file_path)

# Change all the datas into their numeric forms
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Remove those with Open/Close varies that are invalid
data.dropna(subset=['Open', 'Close'], inplace=True)

# PREPARE DATA
# Use the mid-points varies of Open and Close
data['Mid'] = (data['Open'] + data['Close']) / 2
PRICE_VALUE = 'Mid'

# Transfer on all the files to avoid the Out-of-bound error 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1))

# Seperate the Train and Test cards after being transfered
train_data_len = len(data[data.index <= TRAIN_END])
train_scaled = scaled_data[:train_data_len]

x_train, y_train = [], []
for x in range(PREDICTION_DAYS, len(train_scaled)):
    x_train.append(train_scaled[x-PREDICTION_DAYS:x, 0])
    y_train.append(train_scaled[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
# Reshape into (samples, timesteps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# BUILD MODELS
if os.path.exists(MODEL_PATH):
    print("Loading the trained model...")
    model = load_model(MODEL_PATH)
else:
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)
    model.save(MODEL_PATH)

# TESTING ACCURACY
test_scaled = scaled_data[train_data_len - PREDICTION_DAYS:]
actual_prices = data[PRICE_VALUE].values[train_data_len:]

x_test = []
for x in range(PREDICTION_DAYS, len(test_scaled)):
    x_test.append(test_scaled[x-PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


# VISUALIZATON
plt.figure(figsize=(12,6))
plt.plot(actual_prices, color="black", label=f"Actually ({PRICE_VALUE})")
plt.plot(predicted_prices, color="green", label=f"Predicting ({PRICE_VALUE})")

plt.fill_between(range(len(actual_prices)),
    data['Low'].values[train_data_len:],
    data['High'].values[train_data_len:],
    color='gray', alpha=0.2, label='High-Low range')

plt.title(f"Predict stock price {COMPANY} (Task C.2)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()