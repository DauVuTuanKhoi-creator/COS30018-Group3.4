import os
import numpy as np
import pandas as pd
import yfinance as yf 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('vader_lexicon', quiet=True)

COMPANY = 'CBA.AX'
START_DATE = '2023-01-01'
END_DATE = '2024-07-02'

print(f"--- Fetching Stock Data for {COMPANY} ---")

stock_data = yf.download(COMPANY, start=START_DATE, end=END_DATE, group_by='ticker', auto_adjust=True)

if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data = stock_data.copy()
    stock_data.columns = stock_data.columns.get_level_values(-1)

stock_data.dropna(inplace=True)

print(f"--- Fetching & Processing News Data ---")
ticker = yf.Ticker(COMPANY)
news_list = ticker.news

sia = SentimentIntensityAnalyzer()
news_records = []

if news_list:
    for article in news_list:
        title = article.get('title', '')
        pub_date = pd.to_datetime(article.get('providerPublishTime', 0), unit='s').strftime('%Y-%m-%d')

        sentiment = sia.polarity_scores(title)
        compound_score = sentiment['compound']

        news_records.append({'Date': pub_date, 'Sentiment_Score': compound_score})

news_df = pd.DataFrame(news_records)

if not news_df.empty:
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    daily_sentiment = news_df.groupby('Date')['Sentiment_Score'].mean().reset_index()
    daily_sentiment.set_index('Date', inplace=True)
else:
    print("Warning: No news data retrieved. Generating synthetic sentiment for demonstration.")
    np.random.seed(42)
    dates = stock_data.index
    synthetic_scores = np.random.uniform(-1, 1, size=len(dates))
    daily_sentiment = pd.DataFrame({'Sentiment_Score': synthetic_scores}, index=dates)

stock_data.index = pd.to_datetime(stock_data.index).tz_localize(None)
daily_sentiment.index = pd.to_datetime(daily_sentiment.index).tz_localize(None)

merged_data = stock_data.copy()
merged_data = merged_data.join(daily_sentiment, how='left')

merged_data['Sentiment_Score'] = merged_data['Sentiment_Score'].fillna(0)

merged_data['Lag_1_Close'] = merged_data['Close'].shift(1)
merged_data['Price_Charge'] = merged_data['Close'] - merged_data['Lag_1_Close']
merged_data.dropna(inplace=True)

merged_data['Next_Day_Close'] = merged_data['Close'].shift(-1)
merged_data['Target'] = (merged_data['Next_Day_Close'] > merged_data['Close']).astype(int)
merged_data.dropna(inplace=True)

BASE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'Lag_1_Close']
SENTIMENT_FEATURES = BASE_FEATURES + ['Sentiment_Score']

X_base = merged_data[BASE_FEATURES]
X_sentiment = merged_data[SENTIMENT_FEATURES]
y = merged_data['Target']

if len(merged_data) == 0:
    raise ValueError("The data is empty after processing and applying dropna(). Please check the feature columns.")

split_idx = int(len(merged_data) * 0.8)
X_base_train, X_base_test = X_base.iloc[:split_idx], X_base.iloc[split_idx:]
X_sent_train, X_sent_test = X_sentiment.iloc[:split_idx], X_sentiment.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training set: {X_base_train.shape[0]} samples.")
print(f"Test set size: {X_base_test.shape[0]} samples.")

print("\n--- Training Models ---")

baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_base_train, y_train)
base_preds = baseline_model.predict(X_base_test)

sentiment_model = RandomForestClassifier(n_estimators=100, random_state=42)
sentiment_model.fit(X_sent_train, y_train)
sent_preds = sentiment_model.predict(X_sent_test)

def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return cm

cm_base = evaluate_model(y_test, base_preds, "Baseline Model (No Sentiment)")
cm_sent = evaluate_model(y_test, sent_preds, "Enhanced Model (With Sentiment)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Baseline Model Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].set_xticklabels(['Fall (0)', 'Rise (1)'])
axes[0].set_yticklabels(['Fall (0)', 'Rise (1)'])

sns.heatmap(cm_base, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Sentiment-Enhanced Model Confusion Matrix')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')
axes[1].set_xticklabels(['Fall (0)', 'Rise (1)'])
axes[1].set_yticklabels(['Fall (0)', 'Rise (1)'])

plt.tight_layout()
plt.show()