import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import StockChart from './components/StockChart'
import TradingViewWidget from './components/TradingViewWidget'
import { Search, Activity, TrendingUp, DollarSign, BarChart2, Star, ShieldAlert, Target, Award, Plus, Trash2 } from 'lucide-react'

const DEFAULT_WATCHLIST = ['BTC', 'ETH', 'SOL', 'AAPL', 'TSLA', 'NVDA', 'XAUUSD']

function App() {
  const [symbol, setSymbol] = useState('AAPL')
  const [searchInput, setSearchInput] = useState('')
  const [data, setData] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [backtestMetrics, setBacktestMetrics] = useState(null)
  const [predictionRiskLevels, setPredictionRiskLevels] = useState(null)
  const [loading, setLoading] = useState(true)
  const [loadingPred, setLoadingPred] = useState(false)
  const [error, setError] = useState(null)
  const [currentStats, setCurrentStats] = useState(null)
  const [viewMode, setViewMode] = useState('analytics')
  const [selectedInterval, setSelectedInterval] = useState('1d')

  // Request cancellation ref
  const abortControllerRef = useRef(null)

  // Watchlist state backed by LocalStorage
  const [watchlist, setWatchlist] = useState(() => {
    try {
      const saved = localStorage.getItem('future_stocks_watchlist')
      return saved ? JSON.parse(saved) : DEFAULT_WATCHLIST
    } catch (e) {
      return DEFAULT_WATCHLIST
    }
  })

  // Position Sizing Calculator state
  const [accountEquity, setAccountEquity] = useState(10000)
  const [riskPercent, setRiskPercent] = useState(1.0)

  useEffect(() => {
    try {
      localStorage.setItem('future_stocks_watchlist', JSON.stringify(watchlist))
    } catch (e) {
      console.error('Failed to save watchlist:', e)
    }
  }, [watchlist])

  const toggleWatchlist = (sym) => {
    const cleanSym = sym.toUpperCase()
    if (watchlist.includes(cleanSym)) {
      setWatchlist(watchlist.filter(s => s !== cleanSym))
    } else {
      setWatchlist([...watchlist, cleanSym])
    }
  }

  const fetchData = async (sym, intervalVal = selectedInterval) => {
    // Cancel in-flight request if user rapidly changed symbols
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    const controller = new AbortController()
    abortControllerRef.current = controller

    setLoading(true)
    setError(null)
    setPredictions(null)
    setBacktestMetrics(null)
    setPredictionRiskLevels(null)
    try {
      const res = await axios.get(`/api/stock/${sym}?interval=${intervalVal}`, {
        signal: controller.signal
      })
      setData(res.data)
      setCurrentStats(res.data.stats)
      setSymbol(res.data.symbol)

      if (res.data.symbol === 'XAUUSD' || res.data.symbol === 'GC=F' || res.data.symbol.includes('USD')) {
        setViewMode('tradingview')
      }
    } catch (err) {
      if (axios.isCancel(err) || err.name === 'CanceledError') {
        return // Ignore aborted requests
      }
      setError(err.response?.data?.error || "Failed to fetch data. Is the backend running?")
    } finally {
      setLoading(false)
    }
  }

  const fetchCurrentPrice = async () => {
    if (!symbol || document.visibilityState !== 'visible') return
    try {
      const res = await axios.get(`/api/price/${symbol}`)
      if (res.data.price) {
        setCurrentStats(prev => ({
          ...prev,
          close: res.data.price,
          high: Math.max(prev?.high || 0, res.data.price),
          low: Math.min(prev?.low || res.data.price, res.data.price)
        }))
      }
    } catch (err) {
      // Graceful poll error handling
    }
  }

  const fetchPrediction = async (modelType) => {
    if (!data) return
    setLoadingPred(true)
    try {
      const res = await axios.get(`/api/predict/${symbol}?model=${modelType}&interval=${selectedInterval}`)
      setPredictions(res.data.predictions)
      setBacktestMetrics(res.data.backtest_metrics)
      setPredictionRiskLevels(res.data.risk_levels)
    } catch (err) {
      console.error(err)
      alert("Prediction failed: " + (err.response?.data?.error || "Check backend connection"))
    } finally {
      setLoadingPred(false)
    }
  }

  useEffect(() => {
    fetchData('AAPL', '1d')
  }, [])

  // Smart Polling: 10-second interval, paused when browser tab is hidden
  useEffect(() => {
    if (!symbol || !data) return

    let pollInterval = null
    const startPolling = () => {
      if (pollInterval) clearInterval(pollInterval)
      pollInterval = setInterval(() => {
        if (document.visibilityState === 'visible') {
          fetchCurrentPrice()
        }
      }, 10000)
    }

    startPolling()

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        fetchCurrentPrice()
        startPolling()
      } else {
        if (pollInterval) clearInterval(pollInterval)
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => {
      if (pollInterval) clearInterval(pollInterval)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [symbol, data])

  const handleIntervalChange = (newInterval) => {
    setSelectedInterval(newInterval)
    fetchData(symbol, newInterval)
  }

  const handleSearch = (e) => {
    e.preventDefault()
    if (searchInput) {
      fetchData(searchInput, selectedInterval)
      setSearchInput('')
    }
  }

  const latest = data?.data?.[data?.data?.length - 1] || {}
  const previous = data?.data?.[data?.data?.length - 2] || {}

  const livePrice = currentStats?.close ?? latest?.Close ?? 0
  const liveChange = currentStats && latest?.Close ? livePrice - latest.Close : (latest?.Close && previous?.Close ? latest.Close - previous.Close : 0)
  const liveChangePercent = currentStats && latest ? (liveChange / latest.Close) * 100 : (latest && previous ? ((latest.Close - previous.Close) / previous.Close) * 100 : 0)

  // Dynamic Stop Loss from ATR
  const activeRiskLevels = predictionRiskLevels || data?.risk_levels
  const currentATR = activeRiskLevels?.atr || (livePrice * 0.015)
  const stopLossDistance = currentATR * 1.5
  const maxRiskAmount = (accountEquity * (riskPercent / 100.0))
  const suggestedUnits = stopLossDistance > 0 ? (maxRiskAmount / stopLossDistance) : 0
  const positionValue = suggestedUnits * livePrice

  return (
    <div className="container">
      {/* Top Navbar */}
      <div className="header" style={{ marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ background: 'var(--accent)', padding: '0.5rem', borderRadius: '8px', display: 'flex' }}>
            <Activity color="white" size={24} />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: '1.4rem' }}>Future Stock Market</h1>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Quant-AI Forecasting & Risk Intelligence Engine
            </span>
          </div>
        </div>

        {/* Global Search & Interval Controls */}
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
          {/* Interval Selector */}
          <div style={{ display: 'flex', background: 'var(--bg-card)', padding: '0.2rem', borderRadius: '8px', border: '1px solid var(--border)' }}>
            {['1m', '5m', '15m', '1h', '1d'].map((iv) => (
              <button
                key={iv}
                onClick={() => handleIntervalChange(iv)}
                style={{
                  background: selectedInterval === iv ? 'var(--accent)' : 'transparent',
                  color: selectedInterval === iv ? 'white' : 'var(--text-secondary)',
                  border: 'none',
                  padding: '0.25rem 0.6rem',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.75rem',
                  fontWeight: selectedInterval === iv ? 'bold' : 'normal',
                  transition: 'all 0.2s'
                }}
              >
                {iv.toUpperCase()}
              </button>
            ))}
          </div>

          <form onSubmit={handleSearch} className="search-form" style={{ display: 'flex', gap: '0.5rem' }}>
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <Search size={16} style={{ position: 'absolute', left: '10px', color: 'var(--text-secondary)' }} />
              <input
                type="text"
                className="input"
                style={{ paddingLeft: '2rem', width: '180px' }}
                placeholder="Ticker (e.g. BTC, NVDA)..."
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
              />
            </div>
            <button type="submit" className="btn">Search</button>
          </form>
        </div>
      </div>

      {/* Watchlist Bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', overflowX: 'auto', paddingBottom: '0.4rem' }}>
        <span style={{ fontSize: '0.75rem', fontWeight: 'bold', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.3rem', flexShrink: 0 }}>
          <Star size={14} color="#f59e0b" fill="#f59e0b" /> WATCHLIST:
        </span>
        {watchlist.map((fav) => (
          <div
            key={fav}
            style={{
              display: 'flex',
              alignItems: 'center',
              background: symbol === fav ? 'rgba(59, 130, 246, 0.2)' : 'var(--bg-card)',
              border: symbol === fav ? '1px solid var(--accent)' : '1px solid var(--border)',
              borderRadius: '20px',
              padding: '0.2rem 0.6rem',
              gap: '0.4rem',
              cursor: 'pointer',
              flexShrink: 0
            }}
          >
            <span
              onClick={() => fetchData(fav)}
              style={{ fontSize: '0.75rem', fontWeight: '600', color: symbol === fav ? 'var(--accent)' : 'var(--text-primary)' }}
            >
              {fav}
            </span>
            <Trash2
              size={12}
              style={{ opacity: 0.5, cursor: 'pointer' }}
              onClick={(e) => {
                e.stopPropagation()
                toggleWatchlist(fav)
              }}
            />
          </div>
        ))}
        {symbol && !watchlist.includes(symbol) && (
          <button
            onClick={() => toggleWatchlist(symbol)}
            style={{
              background: 'transparent',
              border: '1px dashed var(--border)',
              borderRadius: '20px',
              padding: '0.2rem 0.6rem',
              fontSize: '0.75rem',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.2rem'
            }}
          >
            <Plus size={12} /> Add {symbol}
          </button>
        )}
      </div>

      {error && <div className="error">{error}</div>}

      {loading ? (
        <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-secondary)' }}>
          <Activity className="animate-spin" size={32} style={{ margin: '0 auto 1rem' }} />
          <div>Retrieving market data & computing quantitative indicators...</div>
        </div>
      ) : data ? (
        <>
          {/* Key Metric Header Cards */}
          <div className="grid">
            {/* Live Price Card */}
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <span className="metric-label">{data.company || symbol}</span>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginTop: '0.4rem' }}>
                    <span className="metric-value price-update" style={{ fontSize: '1.8rem', fontWeight: 'bold' }}>
                      ${livePrice.toFixed(2)}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      [{selectedInterval.toUpperCase()}]
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => toggleWatchlist(symbol)}
                  style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: 0 }}
                >
                  <Star size={20} color="#f59e0b" fill={watchlist.includes(symbol) ? '#f59e0b' : 'transparent'} />
                </button>
              </div>
              <div style={{ marginTop: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: liveChange >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                {liveChange >= 0 ? '▲' : '▼'} ${Math.abs(liveChange).toFixed(2)} ({liveChangePercent.toFixed(2)}%)
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>• Source: {data.data_source}</span>
              </div>
            </div>

            {/* Trading Signal Card */}
            <div className="card">
              <span className="metric-label">Signal Recommendation</span>
              <div style={{ marginTop: '0.4rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{
                  fontSize: '1.25rem',
                  fontWeight: '800',
                  color: data.signals?.signal.includes('BUY') ? 'var(--success)' : (data.signals?.signal.includes('SELL') ? 'var(--danger)' : 'var(--text-primary)')
                }}>
                  {data.signals?.signal || 'NEUTRAL'}
                </span>
                <span style={{
                  fontSize: '0.75rem',
                  padding: '0.2rem 0.5rem',
                  borderRadius: '6px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid var(--border)'
                }}>
                  Confidence: {data.signals?.confidence}
                </span>
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                {data.signals?.strategy} • Horiz: {data.signals?.estimated_duration}
              </div>
            </div>

            {/* News Sentiment Impact */}
            {data?.sentiment && (
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="metric-label">News Sentiment (VADER)</span>
                  <span style={{
                    fontSize: '0.72rem',
                    fontWeight: 'bold',
                    padding: '0.15rem 0.5rem',
                    borderRadius: '6px',
                    background: data.sentiment.impact_type === 'positive' ? 'rgba(16, 185, 129, 0.15)' : (data.sentiment.impact_type === 'negative' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(156, 163, 175, 0.15)'),
                    color: data.sentiment.impact_type === 'positive' ? 'var(--success)' : (data.sentiment.impact_type === 'negative' ? 'var(--danger)' : 'var(--text-secondary)')
                  }}>
                    {data.sentiment.label}
                  </span>
                </div>
                <div style={{ fontSize: '1.4rem', fontWeight: 'bold', marginTop: '0.4rem', color: data.sentiment.score > 0 ? 'var(--success)' : (data.sentiment.score < 0 ? 'var(--danger)' : 'var(--text-primary)') }}>
                  {data.sentiment.score > 0 ? '+' : ''}{data.sentiment.score.toFixed(2)}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.4rem' }}>
                  {data.sentiment.impact_direction}
                </div>
              </div>
            )}

            {/* Position Sizing & Risk Management Calculator */}
            <div className="card" style={{ gridColumn: '1 / -1', background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.7))', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <ShieldAlert size={20} color="#3b82f6" />
                  <h3 style={{ margin: 0, fontSize: '1.1rem' }}>Position Sizing & Risk Calculator</h3>
                </div>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                  <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Portfolio Capital ($):
                    <input
                      type="number"
                      value={accountEquity}
                      onChange={(e) => setAccountEquity(Number(e.target.value))}
                      style={{ width: '90px', marginLeft: '0.4rem', padding: '0.2rem 0.4rem', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border)', color: 'white', borderRadius: '4px' }}
                    />
                  </label>
                  <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Risk %:
                    <input
                      type="number"
                      step="0.1"
                      value={riskPercent}
                      onChange={(e) => setRiskPercent(Number(e.target.value))}
                      style={{ width: '60px', marginLeft: '0.4rem', padding: '0.2rem 0.4rem', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border)', color: 'white', borderRadius: '4px' }}
                    />
                  </label>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '1rem' }}>
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.75rem', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Max Risk ($)</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: 'var(--danger)' }}>${maxRiskAmount.toFixed(2)}</div>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.75rem', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Stop Loss (1.5x ATR)</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>${(activeRiskLevels?.stop_loss || (livePrice - stopLossDistance)).toFixed(2)}</div>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.75rem', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Suggested Position Size</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: '#60a5fa' }}>{suggestedUnits.toFixed(2)} Units</div>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.75rem', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Total Capital Allocated</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>${positionValue.toFixed(2)}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Chart View Switcher */}
          <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
            <button
              className="btn"
              style={{ background: viewMode === 'analytics' ? 'var(--accent)' : 'var(--bg-card)', border: viewMode === 'analytics' ? 'none' : '1px solid var(--border)' }}
              onClick={() => setViewMode('analytics')}
            >
              <Activity size={18} style={{ marginRight: '0.5rem', verticalAlign: 'text-bottom' }} />
              Quantitative Analytics & Chart
            </button>
            <button
              className="btn"
              style={{ background: viewMode === 'tradingview' ? 'var(--accent)' : 'var(--bg-card)', border: viewMode === 'tradingview' ? 'none' : '1px solid var(--border)' }}
              onClick={() => setViewMode('tradingview')}
            >
              <BarChart2 size={18} style={{ marginRight: '0.5rem', verticalAlign: 'text-bottom' }} />
              TradingView Live Chart
            </button>
          </div>

          {viewMode === 'analytics' ? (
            <StockChart
              data={data.data}
              predictions={predictions}
              symbol={data.symbol}
              company={data.company}
              riskLevels={activeRiskLevels}
            />
          ) : (
            <TradingViewWidget symbol={data.symbol} />
          )}

          {/* AI Forecast & Backtest Validation Section */}
          <div className="card" style={{ marginTop: '2rem' }}>
            <div className="header" style={{ marginBottom: '1rem', flexWrap: 'wrap', gap: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', flexWrap: 'wrap' }}>
                <h3 style={{ margin: 0 }}>AI Forecast Generator</h3>
                {backtestMetrics && !backtestMetrics.insufficient_data && (
                  <span style={{ fontSize: '0.75rem', background: 'rgba(16, 185, 129, 0.15)', color: 'var(--success)', border: '1px solid rgba(16, 185, 129, 0.3)', padding: '0.2rem 0.6rem', borderRadius: '6px', fontWeight: 'bold' }}>
                    🎯 Walk-Forward Hit Rate: {backtestMetrics.hit_rate_pct}% (Baseline: {backtestMetrics.baseline_majority_hit_rate || 50}%, Edge: {backtestMetrics.edge_over_baseline > 0 ? '+' : ''}{backtestMetrics.edge_over_baseline}%)
                  </span>
                )}
                {backtestMetrics?.insufficient_data && (
                  <span style={{ fontSize: '0.75rem', background: 'rgba(239, 68, 68, 0.1)', color: 'var(--text-secondary)', padding: '0.2rem 0.6rem', borderRadius: '6px' }}>
                    ℹ️ Walk-Forward requires ≥40 historical bars
                  </span>
                )}
              </div>
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                <button className="btn" disabled={loadingPred} onClick={() => fetchPrediction('linear')}>
                  {loadingPred ? 'Thinking...' : 'Linear Momentum Model'}
                </button>
                <button className="btn" disabled={loadingPred} style={{ background: '#8b5cf6' }} onClick={() => fetchPrediction('lstm')}>
                  {loadingPred ? 'Computing Ensemble...' : 'Quant Hybrid Ensemble (MLP+RF+Ridge)'}
                </button>
              </div>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: 0 }}>
              Generates a 7-day direct path return forecast with 95% dynamic EWMA volatility confidence corridors, validated via leak-free walk-forward testing.
            </p>
          </div>

          {/* 7-Day Prediction Table */}
          {predictions && predictions.length > 0 && (
            <div className="card animate-fade-in" style={{ marginTop: '1.5rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <h4 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                  🔮 7-Day Forecast & Dynamic Confidence Corridors
                </h4>
                {backtestMetrics && !backtestMetrics.insufficient_data && (
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                    <span>Model Hit Rate: <strong style={{ color: 'var(--success)' }}>{backtestMetrics.hit_rate_pct}%</strong></span>
                    <span>RMSE: <strong style={{ color: '#60a5fa' }}>{backtestMetrics.rmse}</strong></span>
                    <span>MAPE: <strong style={{ color: '#f59e0b' }}>{backtestMetrics.mape}%</strong></span>
                  </div>
                )}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '1rem' }}>
                {predictions.map((p, idx) => {
                  const isBullish = p.move.includes("Bullish")
                  const badgeColor = isBullish ? 'var(--success)' : 'var(--danger)'
                  const badgeBg = isBullish ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)'

                  return (
                    <div
                      key={idx}
                      style={{
                        background: 'rgba(255, 255, 255, 0.03)',
                        padding: '0.8rem',
                        borderRadius: '10px',
                        border: '1px solid rgba(255, 255, 255, 0.05)',
                        textAlign: 'center',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between',
                        gap: '0.4rem',
                        transition: 'all 0.2s ease',
                      }}
                    >
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: '500' }}>
                        {p.date}
                      </div>
                      <div style={{ fontSize: '1.2rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                        ${p.price.toFixed(2)}
                      </div>
                      {p.upper_95 && (
                        <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)' }}>
                          95%: ${p.lower_95.toFixed(1)} - ${p.upper_95.toFixed(1)}
                        </div>
                      )}
                      <div style={{
                        fontSize: '0.7rem',
                        fontWeight: 'bold',
                        color: badgeColor,
                        background: badgeBg,
                        padding: '0.2rem 0.4rem',
                        borderRadius: '4px',
                        display: 'inline-block',
                        alignSelf: 'center'
                      }}>
                        {isBullish ? '▲ ' : '▼ '}{p.move}
                      </div>
                      <div style={{ fontSize: '0.7rem', color: isBullish ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)', fontWeight: '600' }}>
                        {p.change >= 0 ? '+' : ''}{p.change_percent.toFixed(2)}%
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Academic Evaluation & Methodology Card */}
          <div className="card" style={{ marginTop: '1.5rem', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
            <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.95rem', color: 'var(--accent)' }}>
              🔬 FYP-2 Evaluation & Validation Methodology
            </h4>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Evaluation metrics utilize <strong>TimeSeriesSplit walk-forward validation</strong> with expanding historical windows. Model performance is strictly benchmarked out-of-sample against naive persistence (Random Walk $\hat{y}_{t+1}=y_t$) and majority class drift baselines to eliminate data leakage and survivorship bias.
            </div>
          </div>
        </>
      ) : null}

      {/* Academic Disclaimer Footer */}
      <footer style={{ marginTop: '3rem', textAlign: 'center', padding: '1.5rem', color: 'var(--text-secondary)', fontSize: '0.78rem', borderTop: '1px solid var(--border)' }}>
        <p style={{ margin: 0 }}>
          🎓 <strong>FYP-2 Academic Research Prototype:</strong> Future-Stock-Market Predictive Analytics Platform.
        </p>
        <p style={{ margin: '0.3rem 0 0 0', opacity: 0.7 }}>
          Market forecasting models, risk parameters, and algorithmic indicators are developed strictly for academic evaluation and research demonstration. Not financial advice.
        </p>
      </footer>
    </div>
  )
}

export default App
