# Stock Price Prediction

A full-stack ML application that predicts stock prices using an LSTM neural network.

🔗 **Live Demo:** https://stock-prediction-ashen.vercel.app

## Tech Stack

- **Frontend:** React, Recharts, Axios
- **Backend:** FastAPI, TensorFlow/Keras, yfinance, scikit-learn
- **Deployment:** Vercel (frontend), Render (backend)

## Features

- 🔍 Search any global stock ticker via Yahoo Finance
- 📈 LSTM model trained on 2 years of historical data
- 🔮 30-day price forecast
- 📊 Model metrics: RMSE, MAE, R²
- ⚡ Live price updates every 5 seconds
- 📉 Moving averages (7/30/90-day)
- 💾 Model caching — retrain only every 24 hours per ticker

## ML Model

- 2-layer LSTM (50 units each) with 20% dropout
- 60-day lookback window
- EarlyStopping (patience=5) to prevent overfitting
- MinMaxScaler normalization
- 80/20 train/test split

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/stocks/search` | Search stocks |
| GET | `/api/stocks/{ticker}/data` | Historical OHLCV data |
| GET | `/api/stocks/{ticker}/live` | Live price + change |
| POST | `/api/stocks/predict` | Train model + 30-day forecast |

## Local Setup

**Backend**
```bash
cd backend
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload --port 8001
```

**Frontend**
```bash
cd frontend
npm install
REACT_APP_API_URL=http://localhost:8001/api npm start
```

## Disclaimer

Forecasts are for educational purposes only and not financial advice.
