import React, { useState, useRef } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from "recharts";
import { 
  MagnifyingGlass, TrendUp, Lightning, 
  Quotes, Robot, CalendarBlank, ChartLineUp, Info
} from "@phosphor-icons/react";

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
  const [sentiment, setSentiment] = useState(null);
  const [explanation, setExplanation] = useState(null);
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

    // Fetch sentiment + explanation in parallel
    axios.get(`${API}/stocks/${t}/sentiment`).then(r => setSentiment(r.data)).catch(() => {});
    axios.get(`${API}/stocks/${t}/explain`).then(r => setExplanation(r.data)).catch(() => {});
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
    
    // Connect the lines
    if (historical.length > 0 && predictions.length > 0) {
      predictions[0].actual = historical[historical.length - 1].actual;
    }

    return [...historical, ...predictions];
  };

  const { metrics } = predictionData || {};
  const currentPrice = livePrice?.price || (predictionData?.historical_data && predictionData.historical_data[predictionData.historical_data.length - 1].close) || 0;

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "2rem" }}>
      
      {/* Header & Search */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div style={{ 
            width: 40, height: 40, borderRadius: 8, 
            background: "linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-green) 100%)",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "0 0 20px var(--accent-green-glow)"
          }}>
            <TrendUp weight="bold" size={24} color="#fff" />
          </div>
          <h1 style={{ fontSize: "1.5rem", letterSpacing: "0.05em", color: "var(--text-primary)" }}>AURORA</h1>
        </div>

        <div style={{ display: "flex", gap: "1rem", alignItems: "center", width: "500px" }}>
          <div style={{ flex: 1, position: "relative" }}>
            <MagnifyingGlass size={20} color="var(--text-muted)" style={{ position: "absolute", left: "1rem", top: "50%", transform: "translateY(-50%)" }} />
            <input
              value={query}
              onChange={(e) => handleSearch(e.target.value)}
              placeholder="Search symbols, trends..."
              style={{
                width: "100%", padding: "0.875rem 1rem 0.875rem 2.75rem",
                background: "var(--bg-input)", border: "1px solid var(--border-color)", 
                borderRadius: "0.5rem", color: "var(--text-primary)",
                fontFamily: "IBM Plex Sans", fontSize: "0.875rem", outline: "none",
                transition: "border-color 0.2s"
              }}
              onFocus={(e) => e.target.style.borderColor = "var(--accent-blue)"}
              onBlur={(e) => e.target.style.borderColor = "var(--border-color)"}
            />
            {searchResults.length > 0 && (
              <div style={{
                position: "absolute", top: "100%", left: 0, right: 0, marginTop: "0.5rem",
                background: "var(--bg-card)", border: "1px solid var(--border-color)", 
                borderRadius: "0.5rem", zIndex: 50, overflow: "hidden",
                boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.5)"
              }}>
                {searchResults.map((s) => (
                  <div
                    key={s.ticker}
                    onClick={() => selectStock(s)}
                    style={{ padding: "0.75rem 1rem", cursor: "pointer", borderBottom: "1px solid var(--border-color)", display: "flex", alignItems: "center", gap: "1rem" }}
                    onMouseEnter={(e) => e.currentTarget.style.background = "var(--border-hover)"}
                    onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                  >
                    <span className="font-mono" style={{ color: "var(--accent-blue)", fontWeight: 600 }}>{s.ticker}</span>
                    <span style={{ color: "var(--text-primary)", fontSize: "0.875rem" }}>{s.name}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <button
            onClick={handlePredict}
            disabled={isLoading || !ticker}
            style={{
              padding: "0.875rem 1.5rem", background: isLoading ? "var(--border-color)" : "var(--accent-blue)",
              color: "#fff", border: "none", borderRadius: "0.5rem", cursor: isLoading ? "not-allowed" : "pointer",
              fontFamily: "IBM Plex Sans", fontSize: "0.875rem", fontWeight: 600,
              boxShadow: isLoading ? "none" : "0 4px 14px 0 rgba(59, 130, 246, 0.39)",
              transition: "all 0.2s"
            }}
            onMouseEnter={(e) => !isLoading && (e.currentTarget.style.transform = "translateY(-1px)")}
            onMouseLeave={(e) => !isLoading && (e.currentTarget.style.transform = "none")}
          >
            {isLoading ? "Training..." : "Predict"}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: "1rem", background: "rgba(239, 68, 68, 0.1)", border: "1px solid var(--error-color)", color: "var(--error-color)", borderRadius: "0.5rem", marginBottom: "2rem", display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <Info size={20} />
          {error}
        </div>
      )}

      {/* Progress */}
      {isLoading && (
        <div className="card" style={{ marginBottom: "2rem", display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem", padding: "3rem" }}>
          <div style={{ width: 64, height: 64, borderRadius: "50%", border: "4px solid var(--border-color)", borderTopColor: "var(--accent-green)", animation: "spin 1s linear infinite" }} />
          <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
          <div style={{ width: "100%", maxWidth: 400, background: "var(--bg-input)", borderRadius: 999, height: 8, overflow: "hidden" }}>
             <div style={{ width: `${progress}%`, background: "linear-gradient(90deg, var(--accent-blue), var(--accent-green))", height: "100%", transition: "width 0.3s ease" }} />
          </div>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", fontFamily: "IBM Plex Mono" }}>
             Training LSTM model... {progress}%
          </p>
        </div>
      )}

      {(!isLoading && !predictionData) && (
        <div style={{ height: "400px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: "var(--text-secondary)", gap: "1rem" }}>
           <ChartLineUp size={48} color="var(--border-hover)" />
           <p>Search for a stock and hit Predict to generate analysis.</p>
        </div>
      )}

      {predictionData && (
        <>
          {/* Top Metrics Row */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1.5rem", marginBottom: "1.5rem" }}>
            
            {/* Health / Sentiment Card */}
            <div className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                <h3 style={{ fontSize: "1rem", color: "var(--text-primary)", fontWeight: 500 }}>Live Data</h3>
                <Lightning size={20} color="var(--text-muted)" />
              </div>
              {livePrice ? (
                <>
                  <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem" }}>
                    <span className="font-mono tabular-nums" style={{ fontSize: "2.5rem", fontWeight: 700 }}>${livePrice.price}</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginTop: "1rem" }}>
                    <div>
                      <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>{livePrice.name}</p>
                      <p style={{ fontSize: "0.875rem", color: livePrice.change >= 0 ? "var(--accent-green)" : "var(--error-color)", fontWeight: 600 }}>
                        {livePrice.change >= 0 ? "+" : ""}{livePrice.change} ({livePrice.change_pct}%)
                      </p>
                    </div>
                    <span style={{ 
                      padding: "0.25rem 0.75rem", borderRadius: 999, fontSize: "0.75rem", fontWeight: 700,
                      background: "var(--bullish-bg)", color: "var(--accent-green)", border: "1px solid rgba(16, 185, 129, 0.3)" 
                    }}>
                      ACTIVE
                    </span>
                  </div>
                </>
              ) : (
                <div style={{ color: "var(--text-muted)" }}>Loading live data...</div>
              )}
            </div>

            {/* Prediction Accuracy Card */}
            <div className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                <h3 style={{ fontSize: "1rem", color: "var(--text-primary)", fontWeight: 500 }}>Model Accuracy</h3>
                <TrendUp size={20} color="var(--text-muted)" />
              </div>
              <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem" }}>
                <span className="font-mono tabular-nums" style={{ fontSize: "2.5rem", fontWeight: 700 }}>{(metrics.r2 * 100).toFixed(1)}%</span>
                <span style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>R² Score</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginTop: "1rem" }}>
                <div>
                  <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>MAE: <span style={{color: "var(--text-primary)"}}>{metrics.mae}</span></p>
                  <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>RMSE: <span style={{color: "var(--text-primary)"}}>{metrics.rmse}</span></p>
                </div>
                <span style={{ 
                  padding: "0.25rem 0.75rem", borderRadius: 999, fontSize: "0.75rem", fontWeight: 700,
                  background: "rgba(59, 130, 246, 0.15)", color: "var(--accent-blue)", border: "1px solid rgba(59, 130, 246, 0.3)" 
                }}>
                  HIGH CONFIDENCE
                </span>
              </div>
            </div>

            {/* Sentiment / AI Card */}
            <div className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                <h3 style={{ fontSize: "1rem", color: "var(--text-primary)", fontWeight: 500 }}>Market Sentiment</h3>
                <Quotes size={20} color="var(--text-muted)" />
              </div>
              {sentiment ? (
                <>
                  <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem" }}>
                    <span className="font-mono tabular-nums" style={{ 
                      fontSize: "2.5rem", fontWeight: 700, 
                      color: sentiment.overall === "bullish" ? "var(--bullish-txt)" : sentiment.overall === "bearish" ? "var(--bearish-txt)" : "var(--neutral-txt)"
                    }}>
                      {sentiment.overall.toUpperCase()}
                    </span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginTop: "1rem" }}>
                    <div>
                      <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>Score: {sentiment.score}</p>
                      <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>{sentiment.articles.length} news articles analyzed</p>
                    </div>
                    <span style={{ 
                      padding: "0.25rem 0.75rem", borderRadius: 999, fontSize: "0.75rem", fontWeight: 700,
                      background: sentiment.overall === "bullish" ? "var(--bullish-bg)" : sentiment.overall === "bearish" ? "var(--bearish-bg)" : "var(--neutral-bg)", 
                      color: sentiment.overall === "bullish" ? "var(--bullish-txt)" : sentiment.overall === "bearish" ? "var(--bearish-txt)" : "var(--neutral-txt)",
                      border: `1px solid ${sentiment.overall === "bullish" ? "rgba(16, 185, 129, 0.3)" : sentiment.overall === "bearish" ? "rgba(239, 68, 68, 0.3)" : "rgba(245, 158, 11, 0.3)"}`
                    }}>
                       {sentiment.overall === "bearish" ? "PESSIMISTIC" : sentiment.overall === "bullish" ? "OPTIMISTIC" : "NEUTRAL"}
                    </span>
                  </div>
                </>
              ) : (
                <div style={{ color: "var(--text-muted)" }}>Analyzing news...</div>
              )}
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: "1.5rem" }}>
            {/* Left Column */}
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              
              {/* Chart Card */}
              <div className="card">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
                  <div>
                    <h2 style={{ fontSize: "1.25rem", color: "var(--text-primary)", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      {predictionData.ticker} <span style={{ color: "var(--text-muted)", fontSize: "1rem", fontWeight: 400 }}>| Price History & Projection</span>
                    </h2>
                  </div>
                  <div style={{ display: "flex", gap: "1rem", fontSize: "0.875rem", fontFamily: "IBM Plex Mono" }}>
                     <span style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "var(--text-secondary)" }}>
                       <div style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--accent-blue)" }}/> Historical
                     </span>
                     <span style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "var(--text-secondary)" }}>
                       <div style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--accent-green)", boxShadow: "0 0 8px var(--accent-green)" }}/> Prediction
                     </span>
                  </div>
                </div>

                <div style={{ width: "100%", height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={getChartData()} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                      <XAxis 
                        dataKey="date" 
                        tick={{ fill: "var(--text-secondary)", fontSize: 11, fontFamily: "IBM Plex Mono" }} 
                        axisLine={{ stroke: "var(--border-color)" }}
                        tickLine={{ stroke: "var(--border-color)" }}
                        interval={30} 
                      />
                      <YAxis 
                        tick={{ fill: "var(--text-secondary)", fontSize: 11, fontFamily: "IBM Plex Mono" }} 
                        axisLine={{ stroke: "var(--border-color)" }}
                        tickLine={{ stroke: "var(--border-color)" }}
                        domain={['auto', 'auto']}
                        tickFormatter={(val) => `$${val}`}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: "var(--bg-card)", 
                          borderColor: "var(--border-color)",
                          color: "var(--text-primary)",
                          fontFamily: "IBM Plex Mono",
                          borderRadius: "0.5rem",
                          boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.5)"
                        }} 
                        itemStyle={{ color: "var(--text-primary)" }}
                        labelStyle={{ color: "var(--text-secondary)", marginBottom: "0.5rem" }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="actual" 
                        stroke="var(--accent-blue)" 
                        strokeWidth={2} 
                        dot={false} 
                        name="Historical" 
                      />
                      <Line 
                        type="monotone" 
                        dataKey="predicted" 
                        stroke="var(--accent-green)" 
                        strokeWidth={2} 
                        strokeDasharray="5 5" 
                        dot={false} 
                        name="Predicted" 
                      />
                      {predictionData.predictions.length > 0 && (
                        <ReferenceLine 
                          x={predictionData.predictions[0].date} 
                          stroke="var(--accent-green)" 
                          strokeDasharray="3 3" 
                          opacity={0.5} 
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* AI Explanation Insight */}
              {explanation && (
                <div className="card" style={{ borderLeft: `4px solid ${explanation.outlook === "bullish" ? "var(--accent-green)" : explanation.outlook === "bearish" ? "var(--error-color)" : "var(--neutral-txt)"}`}}>
                  <h3 style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
                    <Robot size={24} color="var(--text-muted)" />
                    AI Market Analysis
                  </h3>
                  <p style={{ color: "var(--text-primary)", lineHeight: 1.6, marginBottom: "1rem" }}>
                    {explanation.summary}
                  </p>
                  <div style={{ background: "var(--bg-input)", borderRadius: "0.5rem", padding: "1rem" }}>
                    <ul style={{ listStyle: "none", padding: 0 }}>
                      {explanation.reasons.map((r, i) => (
                        <li key={i} style={{ display: "flex", alignItems: "flex-start", gap: "0.75rem", marginBottom: "0.5rem", color: "var(--text-secondary)", fontSize: "0.875rem" }}>
                           <div style={{ marginTop: "4px", width: 6, height: 6, borderRadius: "50%", background: "var(--text-muted)", flexShrink: 0 }} />
                           {r}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

            </div>

            {/* Right Column */}
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              
              {/* Forecast Table */}
              <div className="card" style={{ flex: 1 }}>
                 <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                    <h3 style={{ fontSize: "1.1rem" }}>Predicted Trajectory</h3>
                    <CalendarBlank size={20} color="var(--text-muted)" />
                 </div>
                 
                 <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", maxHeight: "550px", overflowY: "auto", paddingRight: "0.5rem" }}>
                   {predictionData.predictions.slice(0, 15).map((p, i) => {
                     const isPositive = currentPrice ? p.predicted_close > currentPrice : true;
                     const diff = currentPrice ? (((p.predicted_close - currentPrice) / currentPrice) * 100).toFixed(2) : 0;
                     
                     return (
                       <div key={p.date} style={{ 
                         display: "flex", justifyContent: "space-between", alignItems: "center", 
                         padding: "0.75rem 1rem", background: "var(--bg-input)", borderRadius: "0.5rem",
                         borderLeft: `2px solid ${isPositive ? "var(--accent-green)" : "var(--error-color)"}`
                       }}>
                          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                            <span style={{ color: "var(--text-secondary)", fontSize: "0.875rem", fontFamily: "IBM Plex Mono", width: "20px" }}>{i+1}</span>
                            <div>
                              <div style={{ color: "var(--text-primary)", fontWeight: 500, fontFamily: "IBM Plex Mono" }}>${p.predicted_close.toFixed(2)}</div>
                              <div style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>{new Date(p.date).toLocaleDateString(undefined, {month: 'short', day: 'numeric'})}</div>
                            </div>
                          </div>
                          <div style={{ 
                            color: isPositive ? "var(--bullish-txt)" : "var(--bearish-txt)", 
                            fontSize: "0.875rem", fontWeight: 600, fontFamily: "IBM Plex Mono" 
                          }}>
                            {isPositive ? "+" : ""}{diff}%
                          </div>
                       </div>
                     )
                   })}
                 </div>
                 
                 {predictionData.predictions.length > 15 && (
                   <p style={{ textAlign: "center", color: "var(--text-muted)", fontSize: "0.75rem", marginTop: "1rem" }}>
                     Showing first 15 days of 30-day forecast
                   </p>
                 )}
              </div>

            </div>
          </div>
        </>
      )}
    </div>
  );
}
