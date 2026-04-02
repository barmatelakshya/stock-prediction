from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import os
import json
from textblob import TextBlob

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
CACHE_HOURS = 24

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


def is_cache_valid(ticker: str) -> bool:
    meta_path = f"{MODELS_DIR}/{ticker}_meta.json"
    if not os.path.exists(meta_path):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    trained_at = datetime.fromisoformat(meta["trained_at"])
    return (datetime.now() - trained_at).total_seconds() < CACHE_HOURS * 3600


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


@app.get("/api/stocks/{ticker}/live")
def get_live_price(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.fast_info
    return {
        "ticker": ticker,
        "price": round(float(info.last_price), 2),
        "change": round(float(info.last_price - info.previous_close), 2),
        "change_pct": round(float((info.last_price - info.previous_close) / info.previous_close * 100), 2),
    }


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
    model_path = f"{MODELS_DIR}/{req.ticker}.keras"

    all_scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(prices.reshape(-1, 1))

    if is_cache_valid(req.ticker) and os.path.exists(model_path):
        model = load_model(model_path)
        # Load scaler params from meta
        with open(f"{MODELS_DIR}/{req.ticker}_meta.json") as f:
            meta = json.load(f)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.array(meta["price_range"]).reshape(-1, 1))
    else:
        train_prices = prices[:split]
        X_train, y_train, scaler = prepare_data(train_prices)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        model = build_lstm_model()
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[es], verbose=0)
        model.save(model_path)

        with open(f"{MODELS_DIR}/{req.ticker}_meta.json", "w") as f:
            json.dump({
                "trained_at": datetime.now().isoformat(),
                "price_range": [float(prices.min()), float(prices.max())]
            }, f)

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
        "cached": is_cache_valid(req.ticker),
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "historical_data": historical,
        "predictions": predictions,
        "technical_indicators": calculate_indicators(df),
    }


@app.get("/api/stocks/{ticker}/sentiment")
def get_sentiment(ticker: str):
    stock = yf.Ticker(ticker)
    news = stock.news[:10] if stock.news else []

    articles = []
    total_polarity = 0
    for item in news:
        title = item.get("content", {}).get("title", "")
        if not title:
            continue
        blob = TextBlob(title)
        polarity = blob.sentiment.polarity
        total_polarity += polarity
        articles.append({
            "title": title,
            "url": item.get("content", {}).get("canonicalUrl", {}).get("url", ""),
            "polarity": round(polarity, 3),
            "label": "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral",
        })

    avg = total_polarity / len(articles) if articles else 0
    overall = "bullish" if avg > 0.1 else "bearish" if avg < -0.1 else "neutral"
    return {"ticker": ticker, "overall": overall, "score": round(avg, 3), "articles": articles}


@app.get("/api/stocks/{ticker}/explain")
def explain(ticker: str):
    df = fetch_stock_data(ticker)
    close = df["Close"]
    volume = df["Volume"]

    price_7d = round(((close.iloc[-1] - close.iloc[-7]) / close.iloc[-7]) * 100, 2)
    price_30d = round(((close.iloc[-1] - close.iloc[-30]) / close.iloc[-30]) * 100, 2)
    vol_change = round(((volume.iloc[-5:].mean() - volume.iloc[-30:-5].mean()) / volume.iloc[-30:-5].mean()) * 100, 2)
    ma7 = close.rolling(7).mean().iloc[-1]
    ma30 = close.rolling(30).mean().iloc[-1]
    above_ma = close.iloc[-1] > ma30

    reasons = []
    if price_7d > 2:
        reasons.append(f"Price up {price_7d}% in the last 7 days — short-term upward momentum")
    elif price_7d < -2:
        reasons.append(f"Price down {abs(price_7d)}% in the last 7 days — short-term selling pressure")

    if price_30d > 5:
        reasons.append(f"Strong 30-day trend: +{price_30d}%")
    elif price_30d < -5:
        reasons.append(f"Weak 30-day trend: {price_30d}%")

    if vol_change > 20:
        reasons.append(f"Volume spike of +{vol_change}% vs 30-day average — increased market interest")
    elif vol_change < -20:
        reasons.append(f"Volume drop of {vol_change}% — declining market interest")

    if above_ma:
        reasons.append(f"Price (${round(float(close.iloc[-1]),2)}) is above 30-day MA (${round(float(ma30),2)}) — bullish signal")
    else:
        reasons.append(f"Price (${round(float(close.iloc[-1]),2)}) is below 30-day MA (${round(float(ma30),2)}) — bearish signal")

    outlook = "bullish" if price_7d > 0 and above_ma else "bearish" if price_7d < 0 and not above_ma else "mixed"

    return {
        "ticker": ticker,
        "outlook": outlook,
        "summary": f"Based on recent price action and volume, the outlook for {ticker} appears {outlook}.",
        "reasons": reasons,
        "stats": {"price_7d": price_7d, "price_30d": price_30d, "vol_change": vol_change},
    }
