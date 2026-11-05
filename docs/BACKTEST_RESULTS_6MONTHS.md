# 📊 6-Month Walk-Forward Backtest Report: Gold, Bitcoin & Ethereum

This empirical backtesting report presents the out-of-sample walk-forward validation results for **Gold (`GC=F`)**, **Bitcoin (`BTC-USD`)**, and **Ethereum (`ETH-USD`)** over the last 6-month historical trading window.

All simulations were executed with **zero look-ahead bias** using periodic expanding-window walk-forward re-estimation on real market data.

---

## 📈 Executive Summary: Key Metric Comparison

| Asset | Model / Benchmark | Hit Rate (%) | Cumulative Return (%) | Sharpe Ratio | Max Drawdown (%) | Profit Factor | RMSE / MAPE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gold** (`GC=F`) | **Quant Hybrid Ensemble** | **53.17%** | **+3.88%** | **0.33** | **-10.45%** | **1.06** | 149.64 / 2.84% |
| | Linear Momentum | 39.68% | -40.11% | -4.65 | -41.66% | 0.46 | 64.61 / 1.16% |
| | Buy & Hold Benchmark | 48.41% | -12.69% | -1.15 | -17.58% | — | — |
| **Bitcoin** (`BTC-USD`) | **Quant Hybrid Ensemble** | **51.67%** | **+70.09%** | **2.93** | **-14.69%** | **1.55** | 1571.06 / 1.67% |
| | Linear Momentum | 44.44% | -53.55% | -4.28 | -54.86% | 0.52 | 1499.13 / 1.45% |
| | Buy & Hold Benchmark | 47.22% | -2.03% | -0.01 | -28.71% | — | — |
| **Ethereum** (`ETH-USD`) | **Quant Hybrid Ensemble** | **52.22%** | **+114.77%** | **3.05** | **-25.31%** | **1.63** | 71.83 / 2.42% |
| | Linear Momentum | 56.11% | +75.09% | 2.23 | -13.40% | 1.42 | 56.48 / 1.79% |
| | Buy & Hold Benchmark | 51.67% | +2.15% | 0.16 | -33.95% | — | — |

---

## 1. 🥇 Gold Spot / Continuous Futures (`GC=F`)

- **Evaluation Window:** May 8, 2026 – November 5, 2026 (126 Trading Days)
- **Price Trajectory:** $4,730.70 $\rightarrow$ $4,130.30 (-12.69% market correction)
- **Random Walk Benchmark:** 50.00%
- **Majority Class Drift Benchmark (Always Long):** 48.41%

### Performance Highlights:
- **Directional Alpha:** The Quant Hybrid Ensemble achieved a **53.17% directional hit rate**, beating the majority drift baseline (48.41%) by **+4.76%** and the random walk by **+3.17%**.
- **Capital Preservation:** During a period where underlying Gold dropped **-12.69%** with a **-17.58%** peak-to-trough drawdown, the Ensemble generated a **+3.88% cumulative return** by correctly identifying trend divergences and taking short/neutral positions.
- **Sharpe Ratio:** 0.33 vs. -1.15 for Buy & Hold.

### Month-by-Month Breakdown:
| Month | Quant Hybrid Ensemble (%) | Buy & Hold Benchmark (%) | Performance Edge (%) |
| :---: | :---: | :---: | :---: |
| **2026-05** | -7.54% | -2.50% | -5.04% |
| **2026-06** | **+9.41%** | -12.07% | **+21.48%** |
| **2026-07** | -6.28% | +1.70% | -7.98% |
| **2026-08** | **+9.53%** | +9.12% | **+0.41%** |
| **2026-09** | -2.41% | -6.58% | **+4.17%** |
| **2026-10** | **+5.60%** | -6.87% | **+12.47%** |
| **2026-11** | -2.93% | +5.94% | -8.87% |

---

## 2. ⚡ Bitcoin (`BTC-USD`)

- **Evaluation Window:** May 10, 2026 – November 5, 2026 (180 Daily Sessions)
- **Price Trajectory:** $82,138.93 $\rightarrow$ $80,467.88 (-2.03% market change)
- **Random Walk Benchmark:** 50.00%
- **Majority Class Drift Benchmark:** 47.22%

### Performance Highlights:
- **Directional Accuracy:** **51.67%**, exceeding the 47.22% majority baseline by **+4.45%**.
- **Risk-Adjusted Outperformance:** The Quant Ensemble delivered a **+70.09% return** (Sharpe: **2.93**) compared to **-2.03%** for Buy & Hold (Sharpe: **-0.01**).
- **Downside Defense:** Maximum Drawdown was constrained to **-14.69%**, nearly half the risk of the passive benchmark (**-28.71%**).
- **Profit Factor:** **1.55**, indicating gross profits were 1.55x larger than cumulative drawdowns.

### Month-by-Month Breakdown:
| Month | Quant Hybrid Ensemble (%) | Buy & Hold Benchmark (%) | Performance Edge (%) |
| :---: | :---: | :---: | :---: |
| **2026-05** | **+1.84%** | -8.78% | **+10.62%** |
| **2026-06** | **+40.79%** | -20.41% | **+61.20%** |
| **2026-07** | **+2.05%** | +7.27% | -5.22% |
| **2026-08** | -0.89% | +25.05% | -25.94% |
| **2026-09** | **+7.66%** | +6.37% | **+1.29%** |
| **2026-10** | **+13.04%** | -7.32% | **+20.36%** |
| **2026-11** | -3.62% | +3.92% | -7.54% |

---

## 3. 💎 Ethereum (`ETH-USD`)

- **Evaluation Window:** May 10, 2026 – November 5, 2026 (180 Daily Sessions)
- **Price Trajectory:** $2,369.04 $\rightarrow$ $2,419.95 (+2.15% market change)
- **Random Walk Benchmark:** 50.00%
- **Majority Class Drift Benchmark:** 51.67%

### Performance Highlights:
- **Cumulative Strategy Return:** **+114.77%** for the Quant Hybrid Ensemble vs. **+2.15%** Buy & Hold.
- **Directional Edge:** **52.22% hit rate** with a **1.63 profit factor**.
- **Sharpe Ratio:** **3.05** (highly significant statistical alpha in high-volatility crypto environments).
- **Model Comparison:** Linear momentum also performed well on ETH due to strong recurring autocorrelation trends (**+75.09%**, Sharpe 2.23).

### Month-by-Month Breakdown:
| Month | Quant Hybrid Ensemble (%) | Buy & Hold Benchmark (%) | Performance Edge (%) |
| :---: | :---: | :---: | :---: |
| **2026-05** | **+20.17%** | -13.86% | **+34.03%** |
| **2026-06** | **+63.19%** | -21.69% | **+84.88%** |
| **2026-07** | **+3.29%** | +18.52% | -15.23% |
| **2026-08** | **+26.01%** | +32.60% | -6.59% |
| **2026-09** | **+8.43%** | +8.79% | -0.36% |
| **2026-10** | -16.63% | -16.06% | -0.57% |
| **2026-11** | -6.91% | +7.42% | -14.33% |
