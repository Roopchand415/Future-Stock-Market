"""
cache.py - High-Performance In-Memory TTL Cache
Thread-safe caching layer with time-to-live expiration to eliminate redundant
network calls and expensive model re-training.
"""

import time
from threading import Lock
from typing import Any, Optional


class TTLCache:
    def __init__(self, default_ttl: int = 60, max_size: int = 500):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            val, expires_at = entry
            if time.time() > expires_at:
                del self._cache[key]
                return None
            return val

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            # Simple eviction if exceeding max_size
            if len(self._cache) >= self.max_size:
                now = time.time()
                # Remove expired entries first
                expired = [k for k, (_, exp) in self._cache.items() if exp < now]
                for k in expired:
                    del self._cache[k]
                # If still too large, drop oldest
                if len(self._cache) >= self.max_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

            expire_time = time.time() + (ttl if ttl is not None else self.default_ttl)
            self._cache[key] = (value, expire_time)

    def delete(self, key: str) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# Pre-configured global cache instances
market_data_cache = TTLCache(default_ttl=60, max_size=300)     # 60 seconds for live OHLCV & price stats
sentiment_cache = TTLCache(default_ttl=900, max_size=200)      # 15 minutes for news sentiment analysis
model_cache = TTLCache(default_ttl=3600, max_size=100)         # 1 hour for trained model artifacts
symbol_lookup_cache = TTLCache(default_ttl=86400, max_size=1000) # 24 hours for stock vs crypto classification
