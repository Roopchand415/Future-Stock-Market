# 📈 Future Stock Market - Quant-AI Financial Forecasting & Risk Intelligence Engine

[![Status](https://img.shields.io/badge/Status-FYP--2%20Active-success)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#)
[![React](https://img.shields.io/badge/React-19-61dafb.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![Tests](https://img.shields.io/badge/Tests-12%20Passed-brightgreen.svg)](#)

A high-performance quantitative financial intelligence and predictive analytics platform engineered for Final Year Project Phase 2 (**FYP-2**). Unifies equities, cryptocurrencies, and spot commodities into a single real-time dashboard featuring **leak-free walk-forward backtesting**, **dynamic volatility confidence corridors**, **VADER financial news sentiment analysis**, and **ATR/Fibonacci risk management**.

---

## 🌟 Key Capabilities & Academic Contributions

### 1. Multi-Asset Market Data Feeds
- **Equities (Stocks)**: Global equities via Yahoo Finance (`AAPL`, `NVDA`, `TSLA`, `MSFT`, `AMZN`, etc.).
- **Cryptocurrencies**: Real-time tick data & klines via Binance API with seamless fallback to CoinGecko and Yahoo Finance.
- **Precious Metals (Gold)**: Spot Gold tracking using Binance `PAXGUSDT` (1:1 physically backed token) and continuous futures (`GC=F` / `XAUUSD=X`), eliminating arbitrary proxy scaling.
- **Multi-Timeframe Analysis**: Intraday to swing trading intervals (`1m`, `5m`, `15m`, `1h`, `1d`).

### 2. Quant-AI Forecasting & Modeling
- **Quant Hybrid Ensemble**: Multi-Layer Perceptron (MLP Neural Net) + Random Forest Regressor + Ridge Regression with exponential recency weighting.
- **Linear Momentum Regression**: Stationary return regression on multi-period lag dynamics and moving average divergence.
- **Direct Multi-Horizon Projections**: Eliminates recursive forecast error compounding by projecting path decay with dynamic volatility scaling.
- **Probabilistic Confidence Corridors**: 95% and 80% forecast cones calculated from Exponentially Weighted Moving Average (EWMA) return volatility.

### 3. Leak-Free Walk-Forward Backtesting
- **Strict Time-Series Split**: Zero look-ahead bias; models train strictly on historical partitions ($t < i$) and evaluate on out-of-sample forward steps ($t = i$).
- **Naive Baseline Benchmarking**: Every model is benchmarked against:
  1. *Persistence Baseline (Random Walk)*: $\hat{y}_{t+1} = y_t$
  2. *Majority Class Drift Baseline*: Always predict "Bullish"
- **Defensible Metrics**: Directional Hit Rate (%), RMSE, MAPE, and Edge-Over-Baseline reported with complete transparency.

### 4. Financial Sentiment & Risk Management
- **VADER Financial NLP**: Context-aware news sentiment engine handling financial idioms (*"all-time high"*, *"earnings beat"*, *"rate cuts"*, *"low unemployment"*), negation inversions, and intensifiers.
- **Dynamic Risk Sizing**: Average True Range (ATR) based Stop-Loss and multi-tier Take-Profit targets (1:1, 1:2, 1:3 Risk/Reward).
- **Fibonacci Retracements**: Automatic detection of swing highs and lows with standard golden-ratio retracement levels (23.6%, 38.2%, 50.0%, 61.8%, 78.6%).
- **Interactive Sizing Calculator**: Computes position size and capital allocation based on portfolio equity and risk tolerance %.

### 5. High-Throughput Engineering
- **Multi-Tier TTL In-Memory Caching**: 60s for market quotes, 15m for sentiment analysis, and 1h for fitted model artifacts.
- **Concurrent Network Execution**: `ThreadPoolExecutor` fetches market data and news feeds in parallel.
- **Smart Client Polling**: 10-second price updates automatically paused when the browser tab is hidden.

---

## 🛠️ Architecture & Tech Stack

- **Frontend**: React 19, Vite, Plotly.js, Lucide Icons, TradingView Embedded Widgets.
- **Backend**: Python Flask 3.1+, Scikit-Learn, SciPy, Pandas, NumPy, yfinance, Gunicorn.
- **Validation & Quality**: Pytest automated test suite (12 tests covering indicators, backtesting, sentiment, and API routes).

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** & **npm**

### 1. Installation

Clone the repository:
```bash
git clone https://github.com/Roopchand415/Future-Stock-Market.git
cd Future-Stock-Market
```

### 2. Backend Setup
```bash
# Install backend dependencies
pip install -r requirements.txt

# Start Flask backend server
python backend/app.py
```
Backend starts on `http://127.0.0.1:5000`.

### 3. Frontend Setup
In a new terminal:
```bash
# Install npm dependencies
npm install

# Start Vite development server
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🧪 Running Automated Tests

Run the test suite with `pytest`:
```bash
python -m pytest tests/ -v
```
All 12 unit tests will execute, validating indicator math, walk-forward splits, sentiment scoring, and API health.

---

## 📂 Project Structure

```text
Future-Stock-Market/
├── backend/
│   ├── app.py                      # Flask REST API & route controllers
│   ├── requirements.txt            # Python dependencies
│   └── services/
│       ├── indicators.py           # Wilder's RSI, ATR, Bollinger, Fibonacci
│       ├── prediction.py           # ML ensemble, walk-forward backtest
│       ├── sentiment.py            # VADER financial sentiment engine
│       ├── cache.py                # Thread-safe in-memory TTLCache
│       ├── binance_api.py          # Binance klines & 24h ticker service
│       └── coingecko.py            # CoinGecko fallback & symbol resolution
├── src/
│   ├── App.jsx                     # Master React dashboard & risk calculator
│   ├── components/
│   │   ├── StockChart.jsx          # Memoized Plotly chart with toggleable layers
│   │   └── TradingViewWidget.jsx   # Embedded TradingView live technical chart
│   └── index.css                   # Responsive dark-theme styling
├── tests/
│   ├── test_indicators.py          # Unit tests for technical indicator math
│   ├── test_backtest.py            # Unit tests for walk-forward validation
│   ├── test_sentiment.py           # Unit tests for VADER financial sentiment
│   └── test_api.py                 # API integration tests
├── docs/                           # Consolidated project documentation & guides
├── Dockerfile                      # Production container definition
├── Procfile                        # Cloud deployment entry point
└── vite.config.js                  # Vite configuration & development proxy
```

---

## 🎓 Academic Disclaimer
This software is developed strictly as an academic research and engineering prototype for **Final Year Project Phase 2 (FYP-2)**. Algorithmic trading signals, probability corridors, and price forecasts are generated for experimental demonstration and educational evaluation only. **This platform does not constitute financial, investment, or legal advice.**
