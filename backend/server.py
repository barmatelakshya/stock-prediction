from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STOCKS = [
    # Tech
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "MSFT", "name": "Microsoft Corporation"},
    {"ticker": "GOOGL", "name": "Alphabet Inc."},
    {"ticker": "AMZN", "name": "Amazon.com Inc."},
    {"ticker": "NVDA", "name": "NVIDIA Corporation"},
    {"ticker": "META", "name": "Meta Platforms Inc."},
    {"ticker": "TSLA", "name": "Tesla Inc."},
    {"ticker": "NFLX", "name": "Netflix Inc."},
    {"ticker": "AMD", "name": "Advanced Micro Devices"},
    {"ticker": "INTC", "name": "Intel Corporation"},
    {"ticker": "ORCL", "name": "Oracle Corporation"},
    {"ticker": "CRM", "name": "Salesforce Inc."},
    {"ticker": "ADBE", "name": "Adobe Inc."},
    {"ticker": "QCOM", "name": "Qualcomm Inc."},
    {"ticker": "TXN", "name": "Texas Instruments"},
    {"ticker": "AVGO", "name": "Broadcom Inc."},
    {"ticker": "IBM", "name": "IBM Corporation"},
    {"ticker": "NOW", "name": "ServiceNow Inc."},
    {"ticker": "SNOW", "name": "Snowflake Inc."},
    {"ticker": "UBER", "name": "Uber Technologies"},
    {"ticker": "LYFT", "name": "Lyft Inc."},
    {"ticker": "SPOT", "name": "Spotify Technology"},
    {"ticker": "SHOP", "name": "Shopify Inc."},
    {"ticker": "SQ", "name": "Block Inc."},
    {"ticker": "PYPL", "name": "PayPal Holdings"},
    {"ticker": "COIN", "name": "Coinbase Global"},
    {"ticker": "PLTR", "name": "Palantir Technologies"},
    {"ticker": "RBLX", "name": "Roblox Corporation"},
    {"ticker": "SNAP", "name": "Snap Inc."},
    {"ticker": "PINS", "name": "Pinterest Inc."},
    {"ticker": "TWLO", "name": "Twilio Inc."},
    {"ticker": "ZM", "name": "Zoom Video Communications"},
    {"ticker": "DOCU", "name": "DocuSign Inc."},
    {"ticker": "OKTA", "name": "Okta Inc."},
    {"ticker": "NET", "name": "Cloudflare Inc."},
    {"ticker": "DDOG", "name": "Datadog Inc."},
    {"ticker": "MDB", "name": "MongoDB Inc."},
    {"ticker": "CRWD", "name": "CrowdStrike Holdings"},
    {"ticker": "ZS", "name": "Zscaler Inc."},
    {"ticker": "PANW", "name": "Palo Alto Networks"},
    # Finance
    {"ticker": "JPM", "name": "JPMorgan Chase"},
    {"ticker": "BAC", "name": "Bank of America"},
    {"ticker": "WFC", "name": "Wells Fargo"},
    {"ticker": "GS", "name": "Goldman Sachs"},
    {"ticker": "MS", "name": "Morgan Stanley"},
    {"ticker": "V", "name": "Visa Inc."},
    {"ticker": "MA", "name": "Mastercard Inc."},
    {"ticker": "AXP", "name": "American Express"},
    {"ticker": "BRK-B", "name": "Berkshire Hathaway"},
    {"ticker": "BLK", "name": "BlackRock Inc."},
    # Healthcare
    {"ticker": "JNJ", "name": "Johnson & Johnson"},
    {"ticker": "PFE", "name": "Pfizer Inc."},
    {"ticker": "MRNA", "name": "Moderna Inc."},
    {"ticker": "ABBV", "name": "AbbVie Inc."},
    {"ticker": "UNH", "name": "UnitedHealth Group"},
    {"ticker": "LLY", "name": "Eli Lilly and Company"},
    {"ticker": "BMY", "name": "Bristol-Myers Squibb"},
    {"ticker": "AMGN", "name": "Amgen Inc."},
    {"ticker": "GILD", "name": "Gilead Sciences"},
    {"ticker": "CVS", "name": "CVS Health"},
    # Consumer
    {"ticker": "WMT", "name": "Walmart Inc."},
    {"ticker": "TGT", "name": "Target Corporation"},
    {"ticker": "COST", "name": "Costco Wholesale"},
    {"ticker": "HD", "name": "Home Depot"},
    {"ticker": "LOW", "name": "Lowe's Companies"},
    {"ticker": "MCD", "name": "McDonald's Corporation"},
    {"ticker": "SBUX", "name": "Starbucks Corporation"},
    {"ticker": "NKE", "name": "Nike Inc."},
    {"ticker": "KO", "name": "Coca-Cola Company"},
    {"ticker": "PEP", "name": "PepsiCo Inc."},
    {"ticker": "PG", "name": "Procter & Gamble"},
    {"ticker": "DIS", "name": "Walt Disney Company"},
    # Energy
    {"ticker": "XOM", "name": "ExxonMobil Corporation"},
    {"ticker": "CVX", "name": "Chevron Corporation"},
    {"ticker": "COP", "name": "ConocoPhillips"},
    {"ticker": "SLB", "name": "SLB (Schlumberger)"},
    # Industrial / Other
    {"ticker": "BA", "name": "Boeing Company"},
    {"ticker": "CAT", "name": "Caterpillar Inc."},
    {"ticker": "GE", "name": "GE Aerospace"},
    {"ticker": "MMM", "name": "3M Company"},
    {"ticker": "HON", "name": "Honeywell International"},
    {"ticker": "UPS", "name": "United Parcel Service"},
    {"ticker": "FDX", "name": "FedEx Corporation"},
    {"ticker": "F", "name": "Ford Motor Company"},
    {"ticker": "GM", "name": "General Motors"},
    {"ticker": "RIVN", "name": "Rivian Automotive"},
    {"ticker": "LCID", "name": "Lucid Group"},
    {"ticker": "SPCE", "name": "Virgin Galactic"},
]

LOOKBACK = 60


class SearchRequest(BaseModel):
    query: str


class PredictRequest(BaseModel):
    ticker: str
    days_ahead: int = 30


def fetch_stock_data(ticker: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    return df


def build_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def prepare_data(prices):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices.reshape(-1, 1))
    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i - LOOKBACK:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler


def calculate_indicators(df):
    close = df["Close"]
    return {
        "ma_7": round(close.rolling(7).mean().iloc[-1], 2),
        "ma_30": round(close.rolling(30).mean().iloc[-1], 2),
        "ma_90": round(close.rolling(90).mean().iloc[-1], 2),
        "avg_volume": int(df["Volume"].mean()),
    }


@app.post("/api/stocks/search")
def search_stocks(req: SearchRequest):
    q = req.query.upper()
    # Local matches first
    local = [s for s in STOCKS if q in s["ticker"] or q in s["name"].upper()]
    if local:
        return {"results": local}
    # Fallback: search Yahoo Finance directly for any stock in the world
    try:
        search = yf.Search(req.query, max_results=10)
        results = []
        for item in search.quotes:
            ticker = item.get("symbol", "")
            name = item.get("longname") or item.get("shortname") or ticker
            if ticker:
                results.append({"ticker": ticker, "name": name})
        return {"results": results}
    except Exception:
        return {"results": []}


@app.get("/api/stocks/{ticker}/data")
def get_stock_data(ticker: str):
    df = fetch_stock_data(ticker)
    data = [
        {"date": str(idx.date()), "open": round(row.Open, 2), "high": round(row.High, 2),
         "low": round(row.Low, 2), "close": round(row.Close, 2), "volume": int(row.Volume)}
        for idx, row in df.iterrows()
    ]
    return {"ticker": ticker, "data": data}


@app.post("/api/stocks/predict")
def predict(req: PredictRequest):
    df = fetch_stock_data(req.ticker)
    prices = df["Close"].values

    split = int(len(prices) * 0.8)
    train_prices = prices[:split]
    test_prices = prices[split:]

    X_train, y_train, scaler = prepare_data(train_prices)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = build_lstm_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Test predictions
    all_scaled = scaler.transform(prices.reshape(-1, 1))
    X_test = []
    for i in range(split, len(all_scaled)):
        X_test.append(all_scaled[i - LOOKBACK:i, 0])
    X_test = np.array(X_test).reshape(-1, LOOKBACK, 1)
    test_preds = scaler.inverse_transform(model.predict(X_test))
    actual = prices[split:]

    rmse = round(float(np.sqrt(mean_squared_error(actual, test_preds))), 4)
    mae = round(float(mean_absolute_error(actual, test_preds)), 4)
    r2 = round(float(r2_score(actual, test_preds)), 4)

    # Future predictions
    last_seq = all_scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    future_preds = []
    seq = last_seq.copy()
    last_date = df.index[-1]
    for i in range(req.days_ahead):
        pred = model.predict(seq, verbose=0)[0][0]
        future_preds.append(pred)
        seq = np.append(seq[:, 1:, :], [[[pred]]], axis=1)

    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    predictions = []
    for i, price in enumerate(future_prices):
        date = last_date + timedelta(days=i + 1)
        predictions.append({"date": str(date.date()), "predicted_close": round(float(price[0]), 2)})

    historical = [
        {"date": str(idx.date()), "close": round(row.Close, 2)}
        for idx, row in df.iterrows()
    ]

    return {
        "ticker": req.ticker,
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "historical_data": historical,
        "predictions": predictions,
        "technical_indicators": calculate_indicators(df),
    }
