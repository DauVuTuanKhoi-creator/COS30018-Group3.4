import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import tensorflow as tf
import yfinance as yf
import mplfinance as mpf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

def plot_candlestick(df, n_days=1):
    """
    Function to display the Candlestick chart.
    Parameters:
        - df (DataFrame): Stock data containing Open, High, Low, Close, Volume columns.
        - n_days (int): Number of trading days to aggregate into a single candle (n>= 1).
    """
    print(f"\n[Task C.3] Plotting Candlestick chart (aggregating {n_days} days)...")

    if n_days > 1:
        group_ids = np.arange(len(df)) // n_days
        df_temp = df.copy()
        df_temp['group_id'] = group_ids

        plot_data = df_temp.groupby('group_id').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })

        plot_data.index = df.index[::n_days][:len(plot_data)]
    else:
        plot_data = df.copy()

    mpf.plot(plot_data[-50:],
             type='candle',
             style='charles',
             title=f'Candlestick Chart ({n_days} days/candle)',
             volume=True,
             ylabel='Price')
    
def plot_boxplot(df, window_size=5, num_windows=15):
    """
    Function to display the Boxplot chart for a sliding window of a n consecutive days,
    Parameters:
        - df (DataFrame): Stock data.
        - window_size (int): Number of consecutive trading days in a window (n).
        - num_windows (int): Limit the number of windows displayed on the chart.
    """
    print(f"\n[Task C.3] Plotting Boxplot (sliding window of {window_size} days)...")

    close_prices = df['Close'].values

    total_possible_windows = len(close_prices) // window_size

    truncated_prices = close_prices[:total_possible_windows * window_size]
    window_data = truncated_prices.reshape((total_possible_windows, window_size))

    display_data = window_data[-num_windows:]

    labels_index = df.index[:total_possible_windows * window_size:window_size][-num_windows:]
    labels = labels_index.strftime('%Y-%m-%d')

    plt.figure(figsize=(12, 6))

    plt.boxplot(display_data.T, vert=True, patch_artist=True, labels=labels)
    plt.title(f'Closing Price Distribution Boxplot (Window: {window_size} days)')
    plt.xlabel('Window Start Date')
    plt.ylabel('Closing Price')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-08-01'
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'
PREDICTION_DAYS = 30

DATA_DIR = 'data'
MODEL_PATH = 'saved_model.h5'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

file_path = os.path.join(DATA_DIR, f"{COMPANY}_data.csv")

if os.path.exists(file_path):
    print(f"Loading data from local file: {file_path}")
    data = pd.read_csv(file_path, header=[0, 1], index_col=0)

    data.columns = data.columns.get_level_values(0)

    if data.index.name != 'Date':
        data.index.name = 'Date'
else:
    print(f"Downloading data from Yahoo Finance...")

    data = yf.download(COMPANY, start=TRAIN_START, end=TEST_END)
    data.to_csv(file_path)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

data.index = pd.to_datetime(data.index)

data.dropna(subset=['Open', 'Close'], inplace=True)

plot_candlestick(data, n_days=1)
plot_candlestick(data, n_days=5)
plot_boxplot(data, window_size=10)

data['Mid'] = (data['Open'] + data['Close']) / 2
PRICE_VALUE = 'Mid'

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1))

train_data_len = len(data[data.index <= TRAIN_END])

train_scaled = scaled_data[:train_data_len]

x_train, y_train = [], []
for x in range(PREDICTION_DAYS, len(train_scaled)):
    x_train.append(train_scaled[x - PREDICTION_DAYS:x, 0])
    y_train.append(train_scaled[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

if os.path.exists(MODEL_PATH):
    print("\nLoading previously trained model...")
    model = load_model(MODEL_PATH)
else:
    print("\nBuilding and training a new model...")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    model.save(MODEL_PATH)
    print("Model saved successfully.")

test_scaled = scaled_data[train_data_len - PREDICTION_DAYS:]
actual_prices = data[PRICE_VALUE].values[train_data_len:]

x_test = []
for x in range(PREDICTION_DAYS, len(test_scaled)):
    x_test.append(test_scaled[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.figure(figsize=(12, 6))
plt.plot(actual_prices, color="black", label=f"Actual {PRICE_VALUE} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {PRICE_VALUE} Price")

plt.fill_between(range(len(actual_prices)),
                 data['Low'].values[train_data_len],
                 data['High'].values[train_data_len],
                 color='gray', alpha=0.3, label='High-Low Fluctuation Band')

plt.title(f"{COMPANY} Stock Price Prediction")
plt.xlabel("Time (Trading Days)")
plt.ylabel(f"{COMPANY} Stock Price")
plt.legend()
plt.show()

real_data = [scaled_data[len(scaled_data) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Predicted value for the next day: {prediction[0][0]}")