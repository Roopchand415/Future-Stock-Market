import os
import sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

import pytest
import numpy as np
import pandas as pd
from services.prediction import train_gru_model, predict_future_gru, train_hybrid_ensemble
from services.model_store import load_model, save_model


def _make_dummy_ohlcv(n=120):
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    base = 100.0 + np.cumsum(np.random.randn(n) * 1.5)
    return pd.DataFrame({
        "Date": dates,
        "Open": base,
        "High": base + np.random.uniform(0.5, 2.0, n),
        "Low": base - np.random.uniform(0.5, 2.0, n),
        "Close": base + np.random.uniform(-0.5, 0.5, n),
        "Volume": np.random.uniform(1000, 5000, n)
    })


def test_gru_training_and_prediction():
    df = _make_dummy_ohlcv(100)
    gru_dict = train_gru_model(df, seq_len=15)
    assert gru_dict is not None
    assert gru_dict.get("is_fallback") is False
    assert "gru" in gru_dict

    preds = predict_future_gru(gru_dict, days=7)
    assert preds is not None
    assert len(preds) == 7
    assert all(isinstance(p, float) and p > 0 for p in preds)


def test_xgboost_ensemble_integration():
    df = _make_dummy_ohlcv(80)
    model_dict, _ = train_hybrid_ensemble(df)
    assert model_dict is not None
    assert model_dict.get("is_fallback") is False
    assert model_dict.get("xgboost_available") is True
    assert model_dict.get("xgb") is not None
