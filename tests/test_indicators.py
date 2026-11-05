import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add backend to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from services.indicators import calculate_indicators, calculate_fibonacci_levels, FEATURE_COLUMNS


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    periods = 100
    dates = pd.date_range('2026-01-01', periods=periods)
    close = 100 + np.cumsum(np.random.randn(periods) * 1.5)
    high = close + np.random.uniform(0.5, 2.0, periods)
    low = close - np.random.uniform(0.5, 2.0, periods)
    open_p = close + np.random.randn(periods) * 0.5
    volume = np.random.randint(10000, 50000, periods)

    return pd.DataFrame({
        'Date': dates,
        'Open': open_p,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })


def test_calculate_indicators_presence(sample_ohlcv):
    df_feat = calculate_indicators(sample_ohlcv, drop_warmup=False)
    
    # Verify core indicator columns exist
    expected = ['RSI', 'ATR', 'EWMA_Vol', 'SMA_20', 'SMA_50', 'EMA_9', 'EMA_21', 'Upper_Band', 'Lower_Band', 'MACD_Line', 'MACD_Signal']
    for col in expected:
        assert col in df_feat.columns, f"Missing expected column: {col}"

    # Verify all ML feature columns exist
    for feat in FEATURE_COLUMNS:
        assert feat in df_feat.columns, f"Missing ML feature: {feat}"


def test_rsi_bounds(sample_ohlcv):
    df_feat = calculate_indicators(sample_ohlcv, drop_warmup=True)
    # Wilder's RSI must always lie within [0, 100]
    assert df_feat['RSI'].min() >= 0.0
    assert df_feat['RSI'].max() <= 100.0


def test_atr_is_not_scalar(sample_ohlcv):
    df_feat = calculate_indicators(sample_ohlcv, drop_warmup=True)
    # ATR must vary across time and not be a static single scalar
    atr_unique_values = df_feat['ATR'].nunique()
    assert atr_unique_values > 10, f"ATR appears static or scalar (only {atr_unique_values} unique values)"


def test_fibonacci_levels(sample_ohlcv):
    fib = calculate_fibonacci_levels(sample_ohlcv)
    assert 'swing_high' in fib
    assert 'swing_low' in fib
    assert fib['swing_high'] >= fib['swing_low']
    assert fib['fib_500'] <= fib['swing_high']
    assert fib['fib_500'] >= fib['swing_low']
