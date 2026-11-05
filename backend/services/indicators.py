"""
indicators.py - Unified Technical Indicators & Feature Engineering Service
Provides mathematically sound, standardized financial indicators across all backend endpoints:
- Wilder's Smoothed RSI (Matching TradingView & Bloomberg standards)
- True Average True Range (ATR) with exponential smoothing
- Moving Averages (SMA 20/50, EMA 9/21, Trend Ratios)
- Bollinger Bands (Upper, Lower, Width, %B)
- MACD (Line, Signal, Histogram)
- Fibonacci Retracement Levels (from swing highs/lows)
- Exponentially Weighted Moving Average (EWMA) Dynamic Volatility
- Proper warm-up row pruning to eliminate cold-start noise
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'Close_to_SMA20', 'Close_to_SMA50', 'EMA9_to_EMA21',
    'RSI', 'MACD_Line', 'MACD_Signal', 'MACD_Diff',
    'BB_Width', 'BB_Percent', 'Return_1d', 'Return_3d',
    'Return_5d', 'ATR_Pct', 'Volume_Ratio'
]


def calculate_indicators(df, drop_warmup=False, warmup_periods=35):
    """
    Calculate full suite of technical indicators on a DataFrame containing
    'Open', 'High', 'Low', 'Close', 'Volume'.
    
    Parameters:
        df (pd.DataFrame): Raw historical OHLCV data.
        drop_warmup (bool): Whether to drop the initial warm-up rows containing un-warmed indicator states.
        warmup_periods (int): Number of initial rows to drop if drop_warmup=True (default: 35).
    
    Returns:
        pd.DataFrame: DataFrame enriched with indicator columns.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Ensure numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1. Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()

    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()

    # Trend & Ratio Features
    df['Close_to_SMA20'] = df['Close'] / (df['SMA_20'] + 1e-9)
    df['Close_to_SMA50'] = df['Close'] / (df['SMA_50'] + 1e-9)
    df['EMA9_to_EMA21'] = df['EMA_9'] / (df['EMA_21'] + 1e-9)

    # 2. RSI with Wilder's Exponential Smoothing (TradingView Standard)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's smoothing: alpha = 1 / 14
    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    df['RSI'] = df['RSI'].fillna(50.0)

    # 3. MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Diff'] = df['MACD_Line'] - df['MACD_Signal']

    # 4. Bollinger Bands (20 periods, 2 std deviations)
    df['SMA20_BB'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['Std20'] = df['Close'].rolling(window=20, min_periods=1).std().fillna(0.0)
    df['Upper_Band'] = df['SMA20_BB'] + (df['Std20'] * 2.0)
    df['Lower_Band'] = df['SMA20_BB'] - (df['Std20'] * 2.0)
    
    bb_range = df['Upper_Band'] - df['Lower_Band']
    df['BB_Width'] = bb_range / (df['SMA20_BB'] + 1e-9)
    df['BB_Percent'] = (df['Close'] - df['Lower_Band']) / (bb_range + 1e-9)
    df['BB_Percent'] = df['BB_Percent'].clip(-0.5, 1.5).fillna(0.5)
    df['BB_Width'] = df['BB_Width'].fillna(0.0)

    # 5. True Range & Wilder's Average True Range (ATR)
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0.0)
    df['ATR'] = df['TR'].ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    # Fallback for early rows
    df['ATR'] = df['ATR'].fillna(df['TR'].rolling(window=14, min_periods=1).mean()).fillna(df['Close'] * 0.01)
    df['ATR_Pct'] = (df['ATR'] / (df['Close'] + 1e-9)).fillna(0.01)

    # 6. Stationary Log Returns
    df['Return_1d'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0.0)
    df['Return_3d'] = np.log(df['Close'] / df['Close'].shift(3)).fillna(0.0)
    df['Return_5d'] = np.log(df['Close'] / df['Close'].shift(5)).fillna(0.0)

    # 7. Dynamic EWMA Volatility (20-bar span)
    df['EWMA_Vol'] = df['Return_1d'].ewm(span=20, min_periods=5).std().fillna(0.012)
    # Ensure floor on volatility so confidence intervals don't collapse
    df['EWMA_Vol'] = df['EWMA_Vol'].clip(lower=0.005)

    # 8. Volume Dynamics
    if 'Volume' in df.columns:
        df['Volume_SMA5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
        df['Volume_Ratio'] = (df['Volume'] / (df['Volume_SMA5'] + 1e-9)).fillna(1.0).clip(0.1, 10.0)
    else:
        df['Volume_Ratio'] = 1.0

    if drop_warmup and len(df) > warmup_periods:
        df = df.iloc[warmup_periods:].copy().reset_index(drop=True)

    return df


def calculate_fibonacci_levels(df, lookback=50):
    """
    Computes key Fibonacci Retracement levels from recent swing high and swing low.
    Levels: 0% (High), 23.6%, 38.2%, 50.0%, 61.8%, 78.6%, 100% (Low).
    """
    if df is None or df.empty:
        return {}

    window = df.tail(lookback)
    swing_high = float(window['High'].max())
    swing_low = float(window['Low'].min())
    diff = swing_high - swing_low

    if diff <= 0:
        current = float(df['Close'].iloc[-1])
        return {
            "swing_high": current * 1.05,
            "swing_low": current * 0.95,
            "fib_236": current * 1.02,
            "fib_382": current * 1.01,
            "fib_500": current,
            "fib_618": current * 0.99,
            "fib_786": current * 0.98
        }

    return {
        "swing_high": round(swing_high, 4),
        "swing_low": round(swing_low, 4),
        "fib_236": round(swing_high - 0.236 * diff, 4),
        "fib_382": round(swing_high - 0.382 * diff, 4),
        "fib_500": round(swing_high - 0.500 * diff, 4),
        "fib_618": round(swing_high - 0.618 * diff, 4),
        "fib_786": round(swing_high - 0.786 * diff, 4)
    }
