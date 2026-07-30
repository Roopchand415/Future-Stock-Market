import { useState, useEffect } from 'react'
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
    setLoading(true)
    setError(null)
    setPredictions(null)
    setBacktestMetrics(null)
    setPredictionRiskLevels(null)
    try {
      const res = await axios.get(`/api/stock/${sym}?interval=${intervalVal}`)
      setData(res.data)
      setCurrentStats(res.data.stats)
      setSymbol(res.data.symbol)

      if (res.data.symbol === 'XAUUSD' || res.data.symbol === 'GC=F' || res.data.symbol.includes('USD')) {
        setViewMode('tradingview')
      }
    } catch (err) {
      setError(err.response?.data?.error || "Failed to fetch data. Is the backend running?")
    } finally {
      setLoading(false)
    }
  }

  const fetchCurrentPrice = async () => {
    if (!symbol) return
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
      console.error('Failed to update price:', err)
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
      alert("Prediction failed. " + (err.response?.data?.error || ""))
    } finally {
      setLoadingPred(false)
    }
  }

  useEffect(() => {
    fetchData('AAPL', '1d')
  }, [])

  useEffect(() => {
    if (!symbol || !data) return
    const interval = setInterval(() => {
      fetchCurrentPrice()
    }, 3000)
    return () => clearInterval(interval)
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

  const [lastUpdated, setLastUpdated] = useState(new Date())
  const [priceFlash, setPriceFlash] = useState(false)

  useEffect(() => {
    if (currentStats) {
      setLastUpdated(new Date())
      setPriceFlash(true)
      setTimeout(() => setPriceFlash(false), 500)
    }
  }, [currentStats])

  // Risk Management & Position Calculations
  const activeRiskLevels = predictionRiskLevels || data?.signals?.risk_levels || data?.risk_levels
  const dollarRisk = (accountEquity * riskPercent) / 100.0
  const riskPerShare = activeRiskLevels?.risk_per_share || (livePrice * 0.015)
  const recommendedUnits = riskPerShare > 0 ? (dollarRisk / riskPerShare) : 0
  const totalPositionValue = recommendedUnits * livePrice

  return (
    <div className="container">
      <header className="header">
        <div>
          <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '800', background: 'linear-gradient(to right, #3b82f6, #818cf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            FutureStocks<span style={{ color: 'white', WebkitTextFillColor: 'white' }}>Prediction</span>
          </h1>
          <p style={{ margin: '0.2rem 0 0 0', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            AI Quantitative Analytics & Risk Management Platform
            {data && (
              <span style={{ marginLeft: '1rem' }}>
                • Last updated: {lastUpdated.toLocaleTimeString()}
              </span>
            )}
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.8rem', alignItems: 'center' }}>
          <select
            value={selectedInterval}
            onChange={(e) => handleIntervalChange(e.target.value)}
            className="input-field"
            style={{
              width: '150px',
              cursor: 'pointer',
              fontWeight: '600',
              paddingLeft: '0.8rem',
              paddingRight: '0.8rem',
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
              borderRadius: '8px'
            }}
          >
            <option value="1m" style={{ background: '#1e293b' }}>1m (Scalping)</option>
            <option value="5m" style={{ background: '#1e293b' }}>5m (Scalping)</option>
            <option value="15m" style={{ background: '#1e293b' }}>15m (Intraday)</option>
            <option value="1h" style={{ background: '#1e293b' }}>1h (Day Trade)</option>
            <option value="1d" style={{ background: '#1e293b' }}>1d (Swing Trade)</option>
          </select>

          <form onSubmit={handleSearch} style={{ position: 'relative', width: '300px' }}>
            <input
              type="text"
              className="input-field"
              placeholder="Search Stock/Crypto (AAPL, BTC, SOL)"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              style={{ paddingLeft: '2.5rem' }}
            />
            <Search size={18} style={{ position: 'absolute', left: '0.8rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
          </form>
        </div>
      </header>

      {/* Quick Watchlist Bar */}
      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
        <span style={{ fontSize: '0.75rem', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Watchlist:
        </span>
        {watchlist.map(sym => (
          <button
            key={sym}
            onClick={() => fetchData(sym, selectedInterval)}
            className={`btn-pill ${symbol === sym ? 'active-blue' : ''}`}
            style={{ fontSize: '0.75rem', padding: '0.2rem 0.6rem' }}
          >
            {sym}
          </button>
        ))}
        {data && (
          <button
            onClick={() => toggleWatchlist(symbol)}
            className="btn-pill"
            style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem', background: watchlist.includes(symbol) ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)' }}
            title={watchlist.includes(symbol) ? "Remove from Watchlist" : "Add to Watchlist"}
          >
            {watchlist.includes(symbol) ? '★ Saved' : '+ Watch'}
          </button>
        )}
      </div>

      {error && (
        <div style={{ color: 'var(--danger)', marginBottom: '1rem', padding: '1rem', border: '1px solid var(--danger)', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.1)' }}>
          {error}
        </div>
      )}

      {data?.warning && (
        <div style={{ color: '#f59e0b', marginBottom: '1rem', padding: '1rem', border: '1px solid #f59e0b', borderRadius: '8px', background: 'rgba(245, 158, 11, 0.1)' }}>
          ⚠️ {data.warning}
        </div>
      )}

      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', margin: '4rem' }}>
          <div className="spinner"></div>
        </div>
      ) : data ? (
        <>
          <div className="grid-dashboard" style={{ marginTop: '0', marginBottom: '1.5rem' }}>
            {/* Live Price Card */}
            <div className={`card flex-center ${priceFlash ? 'price-update' : ''}`} style={{ justifyContent: 'space-between', position: 'relative', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', top: '10px', right: '10px', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--success)', animation: 'pulse 2s infinite' }}></div>
                <span style={{ fontSize: '0.75rem', color: 'var(--success)', fontWeight: '600' }}>LIVE TICKER</span>
              </div>

              <div>
                <div className="metric-label">Current Price</div>
                <div className="metric-value" style={{ fontSize: '2.4rem', background: 'linear-gradient(135deg, #3b82f6, #818cf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                  ${livePrice.toFixed(2)}
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  {data.symbol} - {data.company}
                </div>
              </div>

              <div className="flex-center" style={{ color: liveChange >= 0 ? 'var(--success)' : 'var(--danger)', background: liveChange >= 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)', padding: '0.8rem 1.2rem', borderRadius: '12px', flexDirection: 'column', gap: '0.3rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  {liveChange >= 0 ? <TrendingUp size={24} /> : <TrendingUp size={24} style={{ transform: 'scaleY(-1)' }} />}
                  <span style={{ fontSize: '1.4rem', fontWeight: 'bold' }}>{liveChangePercent.toFixed(2)}%</span>
                </div>
                <span style={{ fontSize: '0.8rem', opacity: 0.8 }}>
                  ${Math.abs(liveChange).toFixed(2)}
                </span>
              </div>
            </div>

            {/* News Sentiment Impact */}
            {data?.sentiment && (
              <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="metric-label" style={{ fontWeight: '600' }}>News Sentiment Effect</span>
                  <span style={{
                    fontSize: '0.75rem',
                    fontWeight: 'bold',
                    padding: '0.15rem 0.5rem',
                    borderRadius: '6px',
                    background: data.sentiment.impact_type === 'positive' ? 'rgba(16, 185, 129, 0.15)' : data.sentiment.impact_type === 'negative' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(156, 163, 175, 0.15)',
                    color: data.sentiment.impact_type === 'positive' ? 'var(--success)' : data.sentiment.impact_type === 'negative' ? 'var(--danger)' : 'var(--text-secondary)'
                  }}>
                    {data.sentiment.impact_direction || 'Neutral'}
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div style={{ fontSize: '1.6rem', fontWeight: '800', color: data.sentiment.impact_type === 'positive' ? 'var(--success)' : data.sentiment.impact_type === 'negative' ? 'var(--danger)' : 'var(--text-primary)', fontFamily: 'monospace' }}>
                    {data.sentiment.next_move_pct > 0 ? '+' : ''}{data.sentiment.next_move_pct?.toFixed(3)}%
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.2' }}>
                    {data.sentiment.predicted_effect || 'Balanced news sentiment.'}
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '80px', overflowY: 'auto', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.4rem' }}>
                  {data.sentiment.headlines && data.sentiment.headlines.length > 0 ? (
                    data.sentiment.headlines.slice(0, 2).map((item, idx) => (
                      <a key={idx} href={item.link} target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none', color: 'inherit', background: 'rgba(255,255,255,0.01)', padding: '0.2rem 0.4rem', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.03)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', gap: '0.5rem' }}>
                          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: '1', WebkitBoxOrient: 'vertical' }}>{item.title}</span>
                          <span style={{ fontWeight: 'bold', color: item.sentiment > 0 ? 'var(--success)' : item.sentiment < 0 ? 'var(--danger)' : 'var(--text-secondary)', flexShrink: 0 }}>{item.sentiment > 0 ? '▲' : item.sentiment < 0 ? '▼' : '●'}</span>
                        </div>
                      </a>
                    ))
                  ) : (
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textAlign: 'center' }}>No recent news</div>
                  )}
                </div>
              </div>
            )}

            {/* Position Sizing & Risk Management Calculator Widget */}
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

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Capital at Risk</div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: 'var(--danger)' }}>${dollarRisk.toFixed(2)}</div>
                </div>

                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Recommended Position Units</div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: '#60a5fa' }}>{recommendedUnits.toFixed(4)} <span style={{ fontSize: '0.8rem' }}>{symbol}</span></div>
                </div>

                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Total Position Value</div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: 'var(--text-primary)' }}>${totalPositionValue.toFixed(2)}</div>
                </div>

                <div style={{ background: 'rgba(239, 68, 68, 0.1)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--danger)' }}>Stop Loss (1.5x ATR)</div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: 'var(--danger)' }}>${activeRiskLevels?.stop_loss ? activeRiskLevels.stop_loss.toFixed(2) : (livePrice * 0.985).toFixed(2)}</div>
                </div>

                <div style={{ background: 'rgba(16, 185, 129, 0.1)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--success)' }}>Take Profit 2 (1:2 R/R)</div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: 'var(--success)' }}>${activeRiskLevels?.take_profit_2 ? activeRiskLevels.take_profit_2.toFixed(2) : (livePrice * 1.03).toFixed(2)}</div>
                </div>
              </div>
            </div>

            {/* Technical Analysis Signals Panel */}
            {data.signals && (
              <div className="card" style={{ gridColumn: '1 / -1' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h3 style={{ margin: 0 }}>Technical Signals & Indicator Breakdown</h3>
                  <div style={{
                    padding: '0.4rem 1rem',
                    borderRadius: '8px',
                    background: data.signals.signal.includes('BUY') ? 'var(--success)' : data.signals.signal.includes('SELL') ? 'var(--danger)' : 'var(--text-secondary)',
                    color: 'white',
                    fontWeight: 'bold',
                    boxShadow: '0 0 15px rgba(0,0,0,0.2)'
                  }}>
                    {data.signals.signal} ({data.signals.confidence})
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
                  <div style={{ background: 'rgba(255,255,255,0.02)', borderRadius: '12px', padding: '1rem' }}>
                    <h4 style={{ marginTop: 0, color: 'var(--text-secondary)' }}>Indicator Analysis</h4>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
                          <th style={{ padding: '0.4rem' }}>Indicator</th>
                          <th style={{ padding: '0.4rem' }}>Value</th>
                          <th style={{ padding: '0.4rem' }}>Signal</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.signals.analysis && data.signals.analysis.map((item, index) => (
                          <tr key={index} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                            <td style={{ padding: '0.6rem 0.4rem' }}>
                              <div style={{ fontWeight: '500' }}>{item.name}</div>
                              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{item.condition}</div>
                            </td>
                            <td style={{ padding: '0.6rem 0.4rem', fontFamily: 'monospace' }}>{item.value}</td>
                            <td style={{ padding: '0.6rem 0.4rem', fontWeight: 'bold', color: item.signal.includes('BUY') ? 'var(--success)' : item.signal.includes('SELL') ? 'var(--danger)' : 'var(--text-secondary)' }}>
                              {item.signal}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Trade Setup Summary */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                    <h4 style={{ marginTop: 0, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span>{data.signals.strategy ? data.signals.strategy : "Trade Strategy"}</span>
                      {data.signals.estimated_duration && (
                        <span style={{ fontSize: '0.75rem', background: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa', padding: '0.15rem 0.5rem', borderRadius: '6px', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                          ⏱️ {data.signals.estimated_duration}
                        </span>
                      )}
                    </h4>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem' }}>
                      <div style={{ background: 'rgba(59, 130, 246, 0.1)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                        <div style={{ color: 'var(--accent)', fontSize: '0.8rem' }}>Entry Price</div>
                        <div style={{ fontSize: '1.15rem', fontWeight: 'bold' }}>${data.signals.entry_price.toFixed(2)}</div>
                      </div>

                      <div style={{ background: 'rgba(16, 185, 129, 0.1)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                        <div style={{ color: 'var(--success)', fontSize: '0.8rem' }}>Take Profit (1:2)</div>
                        <div style={{ fontSize: '1.15rem', fontWeight: 'bold' }}>${data.signals.take_profit.toFixed(2)}</div>
                      </div>

                      <div style={{ background: 'rgba(239, 68, 68, 0.1)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                        <div style={{ color: 'var(--danger)', fontSize: '0.8rem' }}>Stop Loss</div>
                        <div style={{ fontSize: '1.15rem', fontWeight: 'bold' }}>${data.signals.stop_loss.toFixed(2)}</div>
                      </div>

                      <div style={{ background: 'rgba(255, 255, 255, 0.05)', padding: '0.8rem', borderRadius: '8px', border: '1px solid var(--border)' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Consensus Score</div>
                        <div style={{ fontSize: '1.15rem', fontWeight: 'bold' }}>{data.signals.score > 0 ? '+' : ''}{data.signals.score}/10</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
            <button
              className="btn"
              style={{ background: viewMode === 'analytics' ? 'var(--accent)' : 'var(--bg-card)', border: viewMode === 'analytics' ? 'none' : '1px solid var(--border)' }}
              onClick={() => setViewMode('analytics')}
            >
              <Activity size={18} style={{ marginRight: '0.5rem', verticalAlign: 'text-bottom' }} />
              AI Analytics & Chart
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

          {/* AI Prediction Section */}
          <div className="card" style={{ marginTop: '2rem' }}>
            <div className="header" style={{ marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                <h3 style={{ margin: 0 }}>AI Forecast Generator</h3>
                {backtestMetrics && (
                  <span style={{ fontSize: '0.75rem', background: 'rgba(16, 185, 129, 0.15)', color: 'var(--success)', border: '1px solid rgba(16, 185, 129, 0.3)', padding: '0.2rem 0.6rem', borderRadius: '6px', fontWeight: 'bold' }}>
                    🎯 {backtestMetrics.hit_rate_pct}% 30-Day Hit Rate
                  </span>
                )}
              </div>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <button className="btn" disabled={loadingPred} onClick={() => fetchPrediction('linear')}>
                  {loadingPred && predictions?.length === 0 ? 'Thinking...' : 'Linear Regression'}
                </button>
                <button className="btn" disabled={loadingPred} style={{ background: '#8b5cf6' }} onClick={() => fetchPrediction('lstm')}>
                  {loadingPred ? 'Training & Predicting...' : 'Quant-AI Ensemble (LSTM)'}
                </button>
              </div>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              Generate a 7-day stationary return forecast with probabilistic 95% confidence corridors using our machine learning models.
            </p>
          </div>

          {predictions && predictions.length > 0 && (
            <div className="card animate-fade-in" style={{ marginTop: '1.5rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <h4 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                  🔮 7-Day Forecast & Confidence Corridors
                </h4>
                {backtestMetrics && (
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', gap: '1rem' }}>
                    <span>Model Hit Rate: <strong style={{ color: 'var(--success)' }}>{backtestMetrics.hit_rate_pct}%</strong></span>
                    <span>RMSE: <strong style={{ color: '#60a5fa' }}>{backtestMetrics.rmse}</strong></span>
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
                          Range: ${p.lower_95.toFixed(1)} - ${p.upper_95.toFixed(1)}
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
        </>
      ) : null}
    </div>
  )
}

export default App
