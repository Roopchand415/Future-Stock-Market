"""
model_store.py — Persistent Model Caching with joblib

Saves trained models to disk so they don't retrain on every request.
Cache key: (symbol, interval, last_bar_date)
TTL: 24 hours (models are invalidated when new data bar closes)

This takes prediction response time from ~20s → <0.5s for cached symbols.
"""

import os
import time
import hashlib
import joblib
import logging

logger = logging.getLogger(__name__)

# Model store directory (inside backend/)
_STORE_DIR = os.path.join(os.path.dirname(__file__), '..', 'model_cache')
os.makedirs(_STORE_DIR, exist_ok=True)

# TTL: 24 hours in seconds
MODEL_TTL_SECONDS = 86400


def _make_key(symbol: str, interval: str, last_bar_date: str, model_type: str = 'ensemble') -> str:
    """Generate a filesystem-safe cache key."""
    raw = f"{symbol}_{interval}_{last_bar_date}_{model_type}"
    hashed = hashlib.md5(raw.encode()).hexdigest()[:12]
    safe_sym = symbol.replace('/', '_').replace('=', '_')
    return f"{safe_sym}_{interval}_{model_type}_{hashed}"


def _model_path(key: str) -> str:
    return os.path.join(_STORE_DIR, f"{key}.joblib")


def load_model(symbol: str, interval: str, last_bar_date: str, model_type: str = 'ensemble'):
    """
    Load a model from disk if it exists and is fresh (within TTL).
    Returns the model dict, or None if not found / stale.
    """
    key = _make_key(symbol, interval, last_bar_date, model_type)
    path = _model_path(key)

    if not os.path.exists(path):
        return None

    file_age = time.time() - os.path.getmtime(path)
    if file_age > MODEL_TTL_SECONDS:
        try:
            os.remove(path)
        except OSError:
            pass
        logger.info(f"Model stale for {symbol}/{interval} — will retrain")
        return None

    try:
        model = joblib.load(path)
        logger.info(f"Model cache HIT: {symbol}/{interval} ({model_type})")
        return model
    except Exception as e:
        logger.warning(f"Model load failed for {key}: {e}")
        return None


def save_model(model, symbol: str, interval: str, last_bar_date: str, model_type: str = 'ensemble'):
    """
    Persist a model to disk with joblib.
    Skips saving fallback / dummy models.
    """
    if isinstance(model, dict) and model.get('is_fallback', False):
        return  # Don't persist dummy models

    key = _make_key(symbol, interval, last_bar_date, model_type)
    path = _model_path(key)

    try:
        joblib.dump(model, path, compress=3)
        logger.info(f"Model saved: {symbol}/{interval} → {os.path.basename(path)}")
    except Exception as e:
        logger.warning(f"Model save failed for {key}: {e}")


def invalidate_model(symbol: str, interval: str, model_type: str = 'ensemble'):
    """Remove all cached models for a symbol/interval pair."""
    safe_sym = symbol.replace('/', '_').replace('=', '_')
    prefix = f"{safe_sym}_{interval}_{model_type}_"
    removed = 0
    for fname in os.listdir(_STORE_DIR):
        if fname.startswith(prefix):
            try:
                os.remove(os.path.join(_STORE_DIR, fname))
                removed += 1
            except OSError:
                pass
    if removed:
        logger.info(f"Invalidated {removed} cached model(s) for {symbol}/{interval}")


def prune_old_models():
    """Delete all model files older than TTL. Call this at startup."""
    now = time.time()
    pruned = 0
    for fname in os.listdir(_STORE_DIR):
        fpath = os.path.join(_STORE_DIR, fname)
        if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > MODEL_TTL_SECONDS:
            try:
                os.remove(fpath)
                pruned += 1
            except OSError:
                pass
    if pruned:
        logger.info(f"Pruned {pruned} stale model file(s)")
