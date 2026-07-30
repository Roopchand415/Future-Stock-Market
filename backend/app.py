import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder='../dist', static_url_path='/')
CORS(app)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return app.send_static_file(path)
    else:
        return app.send_static_file('index.html')

# --------------------
# Health Check
# --------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Stock Prediction API is running"
    })

# --------------------
# Internal Imports
# --------------------
from services.prediction import (
    train_linear_regression,
    predict_future_linear,
    train_lstm_model,
    predict_future_lstm,
    calculate_trading_signals,
    calculate_backtest_metrics,
    calculate_risk_levels
)

from services.coingecko import (
    is_crypto_symbol,
    fetch_crypto_historical_data,
    fetch_crypto_current_price,
    get_crypto_info
)

from services.sentiment import get_news_sentiment

from services.binance_api import (
    get_binance_klines,
    get_binance_price,
    get_binance_24hr_stats
)

# Optional TradingView import (prevents Render crashes)
try:
    from tradingview_ta import TA_Handler, Interval
    TRADINGVIEW_AVAILABLE = True
except Exception:
    TRADINGVIEW_AVAILABLE = False
    TA_Handler = None
    Interval = None


@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        symbol = symbol.upper()
        original_symbol = symbol
        interval = request.args.get('interval', '1d')
        if interval not in ('1m', '5m', '15m', '1h', '1d'):
            interval = '1d'

        is_crypto = is_crypto_symbol(symbol)
        
        # Mapping of interval to yfinance periods
        yf_periods = {
            '1m': '5d',
            '5m': '1mo',
            '15m': '1mo',
            '1h': '1y',
            '1d': '1y'
        }
        period = yf_periods.get(interval, '1y')

        # Helper to format Datetime/Date column without timezone serialize issues
        def process_dates(df_target):
            if df_target is None or df_target.empty:
                return df_target
            df_target = df_target.copy()
            if 'Date' not in df_target.columns:
                # Find first column that is date/time or index if reset
                for c in df_target.columns:
                    if 'time' in c.lower() or 'date' in c.lower():
                        df_target.rename(columns={c: 'Date'}, inplace=True)
                        break
            if 'Date' not in df_target.columns:
                df_target.rename(columns={df_target.columns[0]: 'Date'}, inplace=True)
            
            df_target['Date'] = pd.to_datetime(df_target['Date'], utc=True)
            df_target['Date'] = df_target['Date'].dt.tz_localize(None) if df_target['Date'].dt.tz is None else df_target['Date'].dt.tz_convert(None)
            
            if interval in ('1m', '5m', '15m', '1h'):
                df_target['Date'] = df_target['Date'].dt.strftime('%Y-%m-%d %H:%M')
            else:
                df_target['Date'] = df_target['Date'].dt.strftime('%Y-%m-%d')
            return df_target

        # =========================
        # CRYPTO / GOLD
        # =========================
        if is_crypto or symbol in ('XAUUSD', 'GOLD'):
            current_data = None
            crypto_info = None
            data_source = None

            # -------- GOLD (XAUUSD) --------
            if symbol in ('XAUUSD', 'GOLD'):
                crypto_info = {'name': 'Gold Spot (XAU/USD)'}

                # Binance PAXG (Primary)
                try:
                    paxg_price = get_binance_price("PAXGUSDT")
                    if paxg_price:
                        current_data = {
                            "price": paxg_price,
                            "change_24h": 0,
                            "volume_24h": 0
                        }
                        data_source = "Binance PAXG"
                except Exception:
                    pass

                # TradingView (Optional)
                if not current_data and TRADINGVIEW_AVAILABLE:
                    tv_sources = [
                        {"exchange": "OANDA", "screener": "forex"},
                        {"exchange": "FX_IDC", "screener": "forex"},
                        {"exchange": "FXCM", "screener": "forex"},
                    ]
                    for cfg in tv_sources:
                        try:
                            handler = TA_Handler(
                                symbol="XAUUSD",
                                screener=cfg["screener"],
                                exchange=cfg["exchange"],
                                interval=Interval.INTERVAL_1_MINUTE
                            )
                            price = handler.get_analysis().indicators.get("close")
                            if price:
                                current_data = {
                                    "price": price,
                                    "change_24h": 0,
                                    "volume_24h": 0
                                }
                                data_source = f"TradingView {cfg['exchange']}"
                                break
                        except Exception:
                            continue

                # yfinance fallback (IAU proxy)
                stock = yf.Ticker("IAU")
                hist = stock.history(period=period, interval=interval)
                if hist.empty:
                    return jsonify({"error": "No gold data available"}), 404

                hist.reset_index(inplace=True)
                scale_factor = 53.4
                for col in ['Open', 'High', 'Low', 'Close']:
                    hist[col] *= scale_factor

                if not current_data:
                    current_data = {
                        "price": float(hist['Close'].iloc[-1]),
                        "change_24h": 0,
                        "volume_24h": 0
                    }
                    data_source = "yfinance IAU (Scaled)"

            # -------- CRYPTO --------
            else:
                stats_24h = get_binance_24hr_stats(symbol)
                hist = get_binance_klines(symbol, interval=interval)

                if hist is not None and not hist.empty:
                    data_source = "Binance API"
                    crypto_info = {"name": f"{symbol}/USDT"}

                    if stats_24h:
                        current_data = {
                            "price": stats_24h["price"],
                            "change_24h": stats_24h["change_24h"],
                            "volume_24h": stats_24h["volume_24h"]
                        }
                else:
                    current_data = fetch_crypto_current_price(symbol)
                    crypto_info = get_crypto_info(symbol)
                    hist = fetch_crypto_historical_data(symbol, days=30 if interval in ('1m', '5m', '15m') else 365)
                    data_source = "CoinGecko"

                # Fallback to yfinance if both Binance and CoinGecko fail (e.g. on Railway/Render due to IP bans/rate-limits)
                if hist is None or hist.empty:
                    try:
                        clean_symbol = symbol.replace('-USD', '').upper().replace('USDT', '')
                        yf_symbol = f"{clean_symbol}-USD"
                        print(f"Binance/CoinGecko failed for {symbol}. Falling back to yfinance with {yf_symbol}...")
                        stock = yf.Ticker(yf_symbol)
                        hist = stock.history(period=period, interval=interval)
                        if not hist.empty:
                            hist.reset_index(inplace=True)
                            data_source = "yfinance (Crypto Fallback)"
                            crypto_info = {"name": f"{clean_symbol}/USD"}
                            
                            # Calculate current price and stats from yahoo finance data
                            latest_close = float(hist['Close'].iloc[-1])
                            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else latest_close
                            change = latest_close - prev_close
                            change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
                            
                            current_data = {
                                "price": latest_close,
                                "change_24h": change_percent,
                                "volume_24h": float(hist['Volume'].iloc[-1]) if len(hist) > 0 else 0
                            }
                    except Exception as e:
                        print(f"yfinance crypto fallback failed for {symbol}: {e}")

            if hist is None or hist.empty:
                return jsonify({"error": "No data found"}), 404

            hist = process_dates(hist)

        # =========================
        # STOCKS
        # =========================
        else:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period, interval=interval)
            if hist.empty:
                return jsonify({"error": "No stock data found"}), 404

            hist.reset_index(inplace=True)
            hist = process_dates(hist)
            data_source = "yfinance"
            crypto_info = {"name": symbol}

        # =========================
        # INDICATORS
        # =========================
        hist['SMA_20'] = hist['Close'].rolling(20).mean()
        hist['SMA_50'] = hist['Close'].rolling(50).mean()

        hist['EMA_9'] = hist['Close'].ewm(span=9, adjust=False).mean()
        hist['EMA_21'] = hist['Close'].ewm(span=21, adjust=False).mean()

        std20 = hist['Close'].rolling(20).std()
        hist['Upper_Band'] = hist['SMA_20'] + (std20 * 2)
        hist['Lower_Band'] = hist['SMA_20'] - (std20 * 2)

        delta = hist['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        hist['RSI'] = 100 - (100 / (1 + rs))

        tr = hist['High'] - hist['Low']
        hist['ATR'] = tr.tail(14).mean()

        df = hist.dropna()
        data = [{
            "Date": r['Date'],
            "Open": float(r['Open']),
            "High": float(r['High']),
            "Low": float(r['Low']),
            "Close": float(r['Close']),
            "Volume": int(r['Volume']),
            "SMA_20": float(r['SMA_20']),
            "SMA_50": float(r['SMA_50']),
            "EMA_9": float(r['EMA_9']),
            "EMA_21": float(r['EMA_21']),
            "Upper_Band": float(r['Upper_Band']),
            "Lower_Band": float(r['Lower_Band']),
            "RSI": float(r['RSI'])
        } for _, r in df.iterrows()]

        latest = df.iloc[-1]
        stats = {
            "open": float(latest['Open']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "close": float(latest['Close']),
            "volume": int(latest['Volume'])
        }

        # Get news sentiment data first so it can be incorporated into trading signals
        try:
            sentiment_data = get_news_sentiment(symbol, is_crypto=is_crypto)
        except Exception as e:
            print(f"Error fetching sentiment: {e}")
            sentiment_data = {
                "score": 0.0,
                "label": "Neutral",
                "headlines": [],
                "impact_type": "neutral",
                "impact_direction": "No Impact",
                "predicted_effect": "Balanced news sentiment indicates neutral impact.",
                "next_move_pct": 0.0,
                "max_impact_pct": 0.0
            }

        signals = calculate_trading_signals(hist, interval=interval, sentiment_data=sentiment_data)
        atr_val = float(latest['ATR']) if 'ATR' in latest and not pd.isna(latest['ATR']) else float(latest['Close']) * 0.01
        risk_levels = calculate_risk_levels(float(latest['Close']), atr_val)

        return jsonify({
            "symbol": symbol,
            "company": crypto_info["name"],
            "data": data,
            "stats": stats,
            "signals": signals,
            "risk_levels": risk_levels,
            "sentiment": sentiment_data,
            "data_source": data_source,
            "warning": None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/price/<symbol>', methods=['GET'])
def get_price_only(symbol):
    try:
        symbol = symbol.upper()
        is_crypto = is_crypto_symbol(symbol)

        if is_crypto or symbol in ('XAUUSD', 'GOLD'):
            if symbol in ('XAUUSD', 'GOLD'):
                try:
                    paxg_price = get_binance_price("PAXGUSDT")
                    if paxg_price:
                        return jsonify({"price": paxg_price})
                except Exception:
                    pass
                
                stock = yf.Ticker("IAU")
                hist = stock.history(period="1d")
                if not hist.empty:
                    return jsonify({"price": float(hist['Close'].iloc[-1]) * 53.4})
                return jsonify({"error": "No price available"}), 404
            else:
                stats_24h = get_binance_24hr_stats(symbol)
                if stats_24h:
                    return jsonify({"price": stats_24h["price"]})
                
                current_data = fetch_crypto_current_price(symbol)
                if current_data:
                    return jsonify({"price": current_data["price"]})
                
                # Fallback to yfinance if both Binance and CoinGecko fail (rate-limits / hosting IP bans)
                try:
                    clean_symbol = symbol.replace('-USD', '').upper().replace('USDT', '')
                    yf_symbol = f"{clean_symbol}-USD"
                    stock = yf.Ticker(yf_symbol)
                    hist = stock.history(period="5d")
                    if not hist.empty:
                        return jsonify({"price": float(hist['Close'].iloc[-1])})
                except Exception:
                    pass
                
                return jsonify({"error": "No price available"}), 404
        else:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="5d")
            if hist.empty:
                return jsonify({"error": "No price available"}), 404
            return jsonify({"price": float(hist['Close'].iloc[-1])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/<symbol>', methods=['GET'])
def predict_stock(symbol):
    try:
        model_type = request.args.get('model', 'linear')
        symbol = symbol.upper()
        interval = request.args.get('interval', '1d')
        if interval not in ('1m', '5m', '15m', '1h', '1d'):
            interval = '1d'
        
        # Mapping of interval to yfinance periods
        yf_periods = {
            '1m': '5d',
            '5m': '1mo',
            '15m': '1mo',
            '1h': '1y',
            '1d': '1y'
        }
        period = yf_periods.get(interval, '1y')

        # Helper to format Datetime/Date column
        def process_dates(df_target):
            if df_target is None or df_target.empty:
                return df_target
            df_target = df_target.copy()
            if 'Date' not in df_target.columns:
                for c in df_target.columns:
                    if 'time' in c.lower() or 'date' in c.lower():
                        df_target.rename(columns={c: 'Date'}, inplace=True)
                        break
            if 'Date' not in df_target.columns:
                df_target.rename(columns={df_target.columns[0]: 'Date'}, inplace=True)
            
            df_target['Date'] = pd.to_datetime(df_target['Date'], utc=True)
            df_target['Date'] = df_target['Date'].dt.tz_localize(None) if df_target['Date'].dt.tz is None else df_target['Date'].dt.tz_convert(None)
            
            if interval in ('1m', '5m', '15m', '1h'):
                df_target['Date'] = df_target['Date'].dt.strftime('%Y-%m-%d %H:%M')
            else:
                df_target['Date'] = df_target['Date'].dt.strftime('%Y-%m-%d')
            return df_target

        hist = None
        
        # Handle XAUUSD/GOLD specially
        if symbol == 'XAUUSD' or symbol == 'GOLD':
            print(f"Fetching gold data for prediction: {symbol}")
            stock = yf.Ticker('IAU')
            try:
                hist = stock.history(period=period, interval=interval)
                if not hist.empty:
                    hist.reset_index(inplace=True)
                    # Scale IAU by 53.4x to get gold spot price
                    hist['Open'] = hist['Open'] * 53.4
                    hist['High'] = hist['High'] * 53.4
                    hist['Low'] = hist['Low'] * 53.4
                    hist['Close'] = hist['Close'] * 53.4
                else:
                    print("IAU returned empty data for prediction")
            except Exception as e:
                print(f"Error fetching IAU for prediction: {e}")
        
        # Use Binance/CoinGecko/yfinance for crypto
        elif is_crypto_symbol(symbol):
            print(f"Fetching crypto data for prediction: {symbol}")
            # Try Binance first
            hist = get_binance_klines(symbol, interval=interval, limit=365)
            
            if hist is None or hist.empty:
                print("Binance prediction fetch failed, trying CoinGecko...")
                hist = fetch_crypto_historical_data(symbol, days=30 if interval in ('1m', '5m', '15m') else 365)
                if hist is not None and not hist.empty:
                    hist.reset_index(drop=True, inplace=True)
            
            if hist is None or hist.empty:
                # Fallback to yfinance if both fail (e.g. on Railway due to IP bans/limits)
                clean_symbol = symbol.replace('-USD', '').upper().replace('USDT', '')
                yf_symbol = f"{clean_symbol}-USD"
                print(f"CoinGecko/Binance prediction fetch failed for {symbol}, falling back to yfinance with {yf_symbol}")
                try:
                    stock = yf.Ticker(yf_symbol)
                    hist = stock.history(period=period, interval=interval)
                    if not hist.empty:
                        hist.reset_index(inplace=True)
                except Exception as e:
                    print(f"yfinance fallback for prediction failed: {e}")

        else:
            # Use yfinance for stocks
            stock = yf.Ticker(symbol)
            try:
                hist = stock.history(period=period, interval=interval)
                if not hist.empty:
                    hist.reset_index(inplace=True)
            except Exception as e:
                print(f"Error fetching data for prediction: {e}")
        
        if hist is None or hist.empty:
            return jsonify({"error": "No data found"}), 404

        hist = process_dates(hist)
            
        # Get News Sentiment to adjust prediction bias
        try:
            sentiment_data = get_news_sentiment(symbol, is_crypto=is_crypto_symbol(symbol))
            sentiment_score = sentiment_data.get("score", 0.0)
            sentiment_label = sentiment_data.get("label", "Neutral")
        except Exception:
            sentiment_score = 0.0
            sentiment_label = "Neutral"

        raw_predictions = []
        sentiment_bias_factor = 0.05 * sentiment_score # up to +/- 5% shift over 7 days based on sentiment
        
        if model_type == 'linear':
            model = train_linear_regression(hist)
            raw_predictions = predict_future_linear(model, hist['Date'].iloc[-1], days=7, interval=interval)
        elif model_type == 'lstm':
            model, scaler = train_lstm_model(hist)
            from services.prediction import prepare_data
            
            look_back = min(60, max(5, len(hist) // 3))
            _, _, _, scaled_data = prepare_data(hist, look_back=look_back)
            preds = predict_future_lstm(model, scaler, scaled_data, look_back=look_back, days=7)
            
            # generating dates
            last_date = hist['Date'].iloc[-1]
            future_dates = []
            for i in range(1, 8):
                if isinstance(last_date, str):
                    last_date_ts = pd.to_datetime(last_date)
                else:
                    last_date_ts = last_date
                
                if interval == '1m':
                    future_dates.append(last_date_ts + pd.Timedelta(minutes=i))
                elif interval == '5m':
                    future_dates.append(last_date_ts + pd.Timedelta(minutes=5*i))
                elif interval == '15m':
                    future_dates.append(last_date_ts + pd.Timedelta(minutes=15*i))
                elif interval == '1h':
                    future_dates.append(last_date_ts + pd.Timedelta(hours=i))
                else:
                    future_dates.append(last_date_ts + pd.Timedelta(days=i))
            
            fmt = '%Y-%m-%d %H:%M' if interval in ('1m', '5m', '15m', '1h') else '%Y-%m-%d'
            raw_predictions = [{"date": d.strftime(fmt), "price": float(p)} for d, p in zip(future_dates, preds)]
            
        # Post-process forecast points with directional sentiment-adjusted move classifications & confidence corridors
        predictions = []
        prev_p = float(hist['Close'].iloc[-1])
        
        hist_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        volatility = float(hist_returns.std()) if len(hist_returns) > 1 else 0.012
        if volatility == 0 or np.isnan(volatility): volatility = 0.012

        for i, p in enumerate(raw_predictions):
            day_index = i + 1
            bias = 1.0 + (sentiment_bias_factor * (day_index / 7.0))
            pred_price = float(p["price"]) * bias
            
            change = pred_price - prev_p
            change_percent = (change / prev_p) * 100 if prev_p != 0 else 0.0
            
            # Confidence Band offsets (80% and 95% confidence corridors)
            band_95_offset = pred_price * (1.96 * volatility * np.sqrt(day_index))
            band_80_offset = pred_price * (1.28 * volatility * np.sqrt(day_index))
            
            upper_95 = pred_price + band_95_offset
            lower_95 = max(0.01, pred_price - band_95_offset)
            upper_80 = pred_price + band_80_offset
            lower_80 = max(0.01, pred_price - band_80_offset)
            
            if change >= 0:
                move = "Strongly Bullish" if sentiment_score >= 0.05 else "Bullish"
            else:
                move = "Strongly Bearish" if sentiment_score <= -0.05 else "Bearish"
                
            predictions.append({
                "date": p["date"],
                "price": pred_price,
                "upper_95": upper_95,
                "lower_95": lower_95,
                "upper_80": upper_80,
                "lower_80": lower_80,
                "change": change,
                "change_percent": change_percent,
                "move": move,
                "sentiment_bias": bias - 1.0,
                "sentiment_label": sentiment_label
            })
            prev_p = pred_price
            
        backtest_metrics = calculate_backtest_metrics(hist, model_type=model_type)
        
        tr = hist['High'] - hist['Low']
        atr_val = float(tr.tail(14).mean()) if len(tr) > 0 else float(hist['Close'].iloc[-1]) * 0.01
        risk_levels = calculate_risk_levels(float(hist['Close'].iloc[-1]), atr_val)

        return jsonify({
            "symbol": symbol,
            "model": model_type,
            "predictions": predictions,
            "backtest_metrics": backtest_metrics,
            "risk_levels": risk_levels
        })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


# =========================
# ENTRY POINT (Render)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
