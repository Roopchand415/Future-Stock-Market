"""
binance_ws.py — Real-Time Binance WebSocket Price Streaming

Provides sub-100ms price updates via Binance's public miniticker WebSocket.
Used by the /api/stream/<symbol> SSE endpoint in app.py.

Architecture:
  - BinanceStreamManager maintains a single persistent WebSocket per symbol
  - Price updates are stored in a thread-safe dict (latest_prices)
  - Flask SSE endpoint reads latest_prices and pushes to connected clients
  - Auto-reconnects on disconnect with exponential backoff
"""

import asyncio
import json
import logging
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Shared in-memory price store: { 'BTCUSDT': {'price': 67234.5, 'ts': 1700000000} }
latest_prices: Dict[str, dict] = {}

# Active WebSocket tasks: { 'BTCUSDT': asyncio.Task }
_active_streams: Dict[str, asyncio.Task] = {}
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None


def _get_or_create_loop():
    """Get or create a persistent background event loop running in a daemon thread."""
    global _loop, _loop_thread

    if _loop is not None and _loop.is_running():
        return _loop

    _loop = asyncio.new_event_loop()

    def _run_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    _loop_thread = threading.Thread(target=_run_loop, args=(_loop,), daemon=True, name="BinanceWS-Loop")
    _loop_thread.start()
    return _loop


async def _stream_price(symbol: str, retries: int = 0):
    """
    Connect to Binance miniTicker WebSocket for a symbol and update latest_prices.
    Reconnects automatically on error with exponential backoff (max 60s).
    """
    binance_symbol = symbol.upper().replace('/', '')
    ws_url = f"wss://stream.binance.com:9443/ws/{binance_symbol.lower()}@miniTicker"

    try:
        import websockets
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
            logger.info(f"[WS] Connected to {binance_symbol} stream")
            retries = 0  # Reset on successful connection

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    # miniTicker format: { c: last_price, h: high, l: low, v: volume, ... }
                    latest_prices[symbol.upper()] = {
                        'price': float(msg.get('c', 0)),
                        'high': float(msg.get('h', 0)),
                        'low': float(msg.get('l', 0)),
                        'volume': float(msg.get('v', 0)),
                        'change_pct': float(msg.get('P', 0)),
                        'ts': time.time()
                    }
                except (json.JSONDecodeError, ValueError):
                    continue

    except Exception as e:
        wait = min(60, 2 ** retries)
        logger.warning(f"[WS] {binance_symbol} stream error: {e} — reconnecting in {wait}s")
        await asyncio.sleep(wait)
        # Schedule reconnection
        asyncio.ensure_future(_stream_price(symbol, retries + 1))


def start_stream(symbol: str):
    """
    Start a WebSocket stream for a Binance symbol (e.g. 'BTC', 'ETH', 'XAUUSD').
    Safe to call multiple times — ignores if stream already active.
    """
    sym_upper = symbol.upper()
    # Map friendly symbols to Binance pairs
    _symbol_map = {
        'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT',
        'BNB': 'BNBUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT',
        'DOGE': 'DOGEUSDT', 'AVAX': 'AVAXUSDT', 'MATIC': 'MATICUSDT',
        'LINK': 'LINKUSDT', 'DOT': 'DOTUSDT', 'UNI': 'UNIUSDT',
        'XAUUSD': 'PAXGUSDT',   # Gold proxy via PAXG
    }
    binance_sym = _symbol_map.get(sym_upper, sym_upper + 'USDT')

    if binance_sym in _active_streams:
        return  # Already streaming

    loop = _get_or_create_loop()
    task = asyncio.run_coroutine_threadsafe(_stream_price(binance_sym), loop)
    _active_streams[binance_sym] = task
    logger.info(f"[WS] Stream started for {binance_sym}")


def get_live_price(symbol: str) -> Optional[dict]:
    """
    Get the latest cached price for a symbol.
    Returns None if no data or data is older than 30 seconds (stream may be down).
    """
    sym_upper = symbol.upper()
    # Try direct and USDT-suffixed key
    for key in [sym_upper, sym_upper + 'USDT', 'PAXGUSDT']:
        data = latest_prices.get(key)
        if data and (time.time() - data.get('ts', 0)) < 30:
            return data
    return None


def stop_stream(symbol: str):
    """Stop and clean up a WebSocket stream."""
    sym_upper = symbol.upper() + 'USDT'
    task = _active_streams.pop(sym_upper, None)
    if task:
        task.cancel()
        logger.info(f"[WS] Stopped stream for {sym_upper}")
