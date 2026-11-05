import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from services.prediction import (
    calculate_backtest_metrics,
    train_hybrid_ensemble,
    predict_future_lstm
)


def test_short_dataframe_returns_insufficient_data():
    df_short = pd.DataFrame({
        'Date': pd.date_range('2026-01-01', periods=20),
        'Open': [100.0] * 20,
        'High': [102.0] * 20,
        'Low': [98.0] * 20,
        'Close': [101.0] * 20,
        'Volume': [5000] * 20
    })

    metrics = calculate_backtest_metrics(df_short)
    assert metrics["insufficient_data"] is True
    assert metrics["hit_rate_pct"] is None
    # Verify fabricated 72.5% is NEVER returned
    assert metrics["hit_rate_pct"] != 72.5


def test_walk_forward_evaluation_authenticity():
    np.random.seed(123)
    periods = 120
    dates = pd.date_range('2026-01-01', periods=periods)
    trend = np.linspace(150, 220, periods) + np.random.randn(periods) * 1.0

    df_test = pd.DataFrame({
        'Date': dates,
        'Open': trend - 0.5,
        'High': trend + 1.5,
        'Low': trend - 1.5,
        'Close': trend,
        'Volume': np.random.randint(20000, 60000, periods)
    })

    metrics = calculate_backtest_metrics(df_test, model_type='lstm')

    assert metrics["insufficient_data"] is False
    assert metrics["total_evaluated"] >= 10
    assert "baseline_persistence_hit_rate" in metrics
    assert "baseline_majority_hit_rate" in metrics
    assert "edge_over_baseline" in metrics
    assert metrics["rmse"] > 0.0
    assert metrics["hit_rate_pct"] != 72.5
