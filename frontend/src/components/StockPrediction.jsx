import React, { useState, useRef } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

const API = process.env.REACT_APP_API_URL || "http://localhost:8001/api";

export default function StockPrediction() {
  const [query, setQuery] = useState("");
  const [ticker, setTicker] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [predictionData, setPredictionData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [livePrice, setLivePrice] = useState(null);
  const progressRef = useRef(null);
  const liveRef = useRef(null);

  const handleSearch = async (q) => {
    setQuery(q);
    setTicker(q);
    if (!q) return setSearchResults([]);
    const res = await axios.post(`${API}/stocks/search`, { query: q });
    setSearchResults(res.data.results);
  };

  const selectStock = (stock) => {
    setQuery(stock.name);
    setTicker(stock.ticker);
    setSearchResults([]);
    startLivePoll(stock.ticker);
  };

  const startLivePoll = (t) => {
    clearInterval(liveRef.current);
    const fetchLive = async () => {
      try {
        const res = await axios.get(`${API}/stocks/${t}/live`);
        setLivePrice(res.data);
      } catch {}
    };
    fetchLive();
    liveRef.current = setInterval(fetchLive, 5000);
  };

  const handlePredict = async () => {
    if (!ticker) return;
    setIsLoading(true);
    setProgress(0);
    setError("");
    setPredictionData(null);

    progressRef.current = setInterval(() => {
      setProgress((p) => (p < 90 ? p + 5 : p));
    }, 600);

    try {
      const res = await axios.post(`${API}/stocks/predict`, {
        ticker: ticker.toUpperCase(),
        days_ahead: 30,
      });
      setPredictionData(res.data);
      setProgress(100);
    } catch {
      setError("Prediction failed. Check the ticker and try again.");
    } finally {
      clearInterval(progressRef.current);
      setIsLoading(false);
    }
  };

  const getChartData = () => {
    if (!predictionData) return [];
    const historical = predictionData.historical_data.slice(-120).map((d) => ({
      date: d.date,
      actual: d.close,
      predicted: null,
    }));
    const predictions = predictionData.predictions.map((p) => ({
      date: p.date,
      actual: null,
      predicted: p.predicted_close,
    }));
    return [...historical, ...predictions];
  };

  const { metrics, technical_indicators: ti } = predictionData || {};

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "2rem 1rem" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "2rem" }}>Stock Price Prediction</h1>

      {/* Search */}
      <div style={{ position: "relative", display: "flex", gap: "1rem", marginBottom: "1.5rem" }}>
        <div style={{ flex: 1, position: "relative" }}>
          <input
            value={query}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search stock (e.g. AAPL, Tesla)"
            style={{
              width: "100%", padding: "0.75rem 1rem",
              border: "1px solid #E5E7EB", fontFamily: "IBM Plex Sans", fontSize: "1rem",
              outline: "none", background: "#fff"
            }}
          />
          {searchResults.length > 0 && (
            <div style={{
              position: "absolute", top: "100%", left: 0, right: 0,
              background: "#fff", border: "1px solid #E5E7EB", zIndex: 10
            }}>
              {searchResults.map((s) => (
                <div
                  key={s.ticker}
                  onClick={() => selectStock(s)}
                  style={{ padding: "0.75rem 1rem", cursor: "pointer", borderBottom: "1px solid #F3F4F6" }}
                  onMouseEnter={(e) => e.target.style.background = "#F8F9FA"}
                  onMouseLeave={(e) => e.target.style.background = "#fff"}
                >
                  <span className="font-mono" style={{ marginRight: "0.75rem", fontWeight: 600 }}>{s.ticker}</span>
                  {s.name}
                </div>
              ))}
            </div>
          )}
        </div>
        <button
          onClick={handlePredict}
          disabled={isLoading || !ticker}
          style={{
            padding: "0.75rem 2rem", background: isLoading ? "#9CA3AF" : "#111827",
            color: "#fff", border: "none", cursor: isLoading ? "not-allowed" : "pointer",
            fontFamily: "IBM Plex Sans", fontSize: "1rem", fontWeight: 600
          }}
        >
          {isLoading ? "Training..." : "Predict"}
        </button>
      </div>

      {/* Live Price */}
      {livePrice && (
        <div className="card" style={{ display: "inline-flex", alignItems: "center", gap: "1.5rem", marginBottom: "1.5rem", padding: "1rem 1.5rem" }}>
          <span className="font-mono" style={{ fontWeight: 700, fontSize: "1.1rem" }}>{livePrice.ticker}</span>
          <span className="font-mono tabular-nums" style={{ fontSize: "1.5rem", fontWeight: 700 }}>${livePrice.price}</span>
          <span className="font-mono tabular-nums" style={{
            fontSize: "1rem", fontWeight: 600,
            color: livePrice.change >= 0 ? "#16A34A" : "#DC2626"
          }}>
            {livePrice.change >= 0 ? "▲" : "▼"} {Math.abs(livePrice.change)} ({Math.abs(livePrice.change_pct)}%)
          </span>
          <span style={{ fontSize: "0.75rem", color: "#9CA3AF" }}>live · updates every 5s</span>
        </div>
      )}

      {/* Progress */}
      {isLoading && (
        <div style={{ marginBottom: "1.5rem" }}>
          <div style={{ background: "#E5E7EB", height: 6 }}>
            <div style={{ width: `${progress}%`, background: "#111827", height: "100%", transition: "width 0.3s" }} />
          </div>
          <p style={{ marginTop: "0.5rem", fontSize: "0.875rem", color: "#6B7280" }}>
            Training LSTM model... {progress}%
          </p>
        </div>
      )}

      {error && <p style={{ color: "#DC2626", marginBottom: "1rem" }}>{error}</p>}

      {predictionData && (
        <>
          {/* Metrics */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "1.5rem" }}>
            {[
              { label: "Ticker", value: predictionData.ticker },
              { label: "RMSE", value: metrics.rmse },
              { label: "MAE", value: metrics.mae },
              { label: "R² Score", value: metrics.r2 },
            ].map(({ label, value }) => (
              <div key={label} className="card">
                <p style={{ fontSize: "0.75rem", color: "#6B7280", marginBottom: "0.5rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</p>
                <p className="font-mono tabular-nums" style={{ fontSize: "1.5rem", fontWeight: 700 }}>{value}</p>
              </div>
            ))}
          </div>

          {/* Chart */}
          <div className="card" style={{ marginBottom: "1.5rem" }}>
            <h3 style={{ marginBottom: "1rem" }}>Price Chart</h3>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={getChartData()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis dataKey="date" tick={{ fontFamily: "IBM Plex Mono", fontSize: 11 }} interval={20} />
                <YAxis tick={{ fontFamily: "IBM Plex Mono", fontSize: 11 }} />
                <Tooltip contentStyle={{ fontFamily: "IBM Plex Mono", fontSize: 12 }} />
                <Legend />
                <Line dataKey="actual" stroke="#111827" strokeWidth={2} dot={false} name="Historical Price" connectNulls={false} />
                <Line dataKey="predicted" stroke="#2563EB" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Predicted Price" connectNulls={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Technical Indicators */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "1.5rem" }}>
            {[
              { label: "7-Day MA", value: ti.ma_7 },
              { label: "30-Day MA", value: ti.ma_30 },
              { label: "90-Day MA", value: ti.ma_90 },
              { label: "Avg Volume", value: ti.avg_volume.toLocaleString() },
            ].map(({ label, value }) => (
              <div key={label} className="card">
                <p style={{ fontSize: "0.75rem", color: "#6B7280", marginBottom: "0.5rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</p>
                <p className="font-mono tabular-nums" style={{ fontSize: "1.25rem", fontWeight: 600 }}>{value}</p>
              </div>
            ))}
          </div>

          {/* Forecast Table */}
          <div className="card">
            <h3 style={{ marginBottom: "1rem" }}>30-Day Forecast</h3>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "IBM Plex Mono", fontSize: "0.875rem" }}>
              <thead>
                <tr style={{ borderBottom: "2px solid #E5E7EB" }}>
                  <th style={{ textAlign: "left", padding: "0.5rem 1rem" }}>Date</th>
                  <th style={{ textAlign: "right", padding: "0.5rem 1rem" }}>Predicted Close</th>
                </tr>
              </thead>
              <tbody>
                {predictionData.predictions.map((p) => (
                  <tr key={p.date} style={{ borderBottom: "1px solid #F3F4F6" }}>
                    <td style={{ padding: "0.5rem 1rem" }}>{p.date}</td>
                    <td style={{ padding: "0.5rem 1rem", textAlign: "right" }}>${p.predicted_close}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
