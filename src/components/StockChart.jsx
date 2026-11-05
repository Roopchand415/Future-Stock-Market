import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';

const StockChart = ({ data, predictions, symbol, company, riskLevels }) => {
    if (!data || data.length === 0) return null;

    const [showSMA, setShowSMA] = useState(true);
    const [showEMA, setShowEMA] = useState(false);
    const [showBB, setShowBB] = useState(false);
    const [showConfidence, setShowConfidence] = useState(true);
    const [showRiskLevels, setShowRiskLevels] = useState(true);
    const [showFibonacci, setShowFibonacci] = useState(false);

    const { dates, close, sma20, sma50, ema9, ema21, upperBB, lowerBB } = useMemo(() => {
        return {
            dates: data.map(d => d.Date),
            close: data.map(d => d.Close),
            sma20: data.map(d => d.SMA_20),
            sma50: data.map(d => d.SMA_50),
            ema9: data.map(d => d.EMA_9),
            ema21: data.map(d => d.EMA_21),
            upperBB: data.map(d => d.Upper_Band),
            lowerBB: data.map(d => d.Lower_Band)
        };
    }, [data]);

    const traces = useMemo(() => {
        const trs = [
            {
                x: dates,
                y: close,
                type: 'scatter',
                mode: 'lines',
                name: 'Close Price',
                line: { color: '#3b82f6', width: 2.5 }
            }
        ];

        // Simple Moving Averages
        if (showSMA) {
            if (sma20 && sma20.some(v => v !== undefined)) {
                trs.push({
                    x: dates,
                    y: sma20,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'SMA 20',
                    line: { color: '#10b981', width: 1.5 },
                    opacity: 0.8
                });
            }
            if (sma50 && sma50.some(v => v !== undefined)) {
                trs.push({
                    x: dates,
                    y: sma50,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'SMA 50',
                    line: { color: '#f59e0b', width: 1.5 },
                    opacity: 0.8
                });
            }
        }

        // Exponential Moving Averages
        if (showEMA) {
            if (ema9 && ema9.some(v => v !== undefined)) {
                trs.push({
                    x: dates,
                    y: ema9,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'EMA 9',
                    line: { color: '#ec4899', width: 1.5, dash: 'dot' },
                    opacity: 0.8
                });
            }
            if (ema21 && ema21.some(v => v !== undefined)) {
                trs.push({
                    x: dates,
                    y: ema21,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'EMA 21',
                    line: { color: '#8b5cf6', width: 1.5, dash: 'dot' },
                    opacity: 0.8
                });
            }
        }

        // Bollinger Bands
        if (showBB && upperBB && lowerBB && upperBB.some(v => v !== undefined)) {
            trs.push({
                x: dates,
                y: upperBB,
                type: 'scatter',
                mode: 'lines',
                name: 'Upper BB',
                line: { color: 'rgba(156, 163, 175, 0.4)', width: 1 },
                showlegend: false
            });
            trs.push({
                x: dates,
                y: lowerBB,
                type: 'scatter',
                mode: 'lines',
                name: 'Bollinger Bands',
                line: { color: 'rgba(156, 163, 175, 0.4)', width: 1 },
                fill: 'tonexty',
                fillcolor: 'rgba(156, 163, 175, 0.05)'
            });
        }

        // AI Predictions and Dynamic Confidence Corridors
        if (predictions && predictions.length > 0) {
            const predDates = predictions.map(p => p.date);
            const predPrices = predictions.map(p => p.price);

            // Connect last historical point to first prediction
            const lastDate = dates[dates.length - 1];
            const lastClose = close[close.length - 1];

            const forecastDates = [lastDate, ...predDates];
            const forecastPrices = [lastClose, ...predPrices];

            if (showConfidence && predictions[0].upper_95 !== undefined) {
                const upper95 = [lastClose, ...predictions.map(p => p.upper_95)];
                const lower95 = [lastClose, ...predictions.map(p => p.lower_95)];

                trs.push({
                    x: forecastDates,
                    y: upper95,
                    type: 'scatter',
                    mode: 'lines',
                    name: '95% Upper Bound',
                    line: { color: 'rgba(139, 92, 246, 0.2)', width: 0.5 },
                    showlegend: false
                });

                trs.push({
                    x: forecastDates,
                    y: lower95,
                    type: 'scatter',
                    mode: 'lines',
                    name: '95% Confidence Corridor',
                    fill: 'tonexty',
                    fillcolor: 'rgba(139, 92, 246, 0.12)',
                    line: { color: 'rgba(139, 92, 246, 0.2)', width: 0.5 }
                });
            }

            trs.push({
                x: forecastDates,
                y: forecastPrices,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'AI Forecast',
                line: { color: '#a855f7', width: 2.5, dash: 'dash' },
                marker: { size: 6, color: '#c084fc' }
            });
        }

        // ATR Risk Management Levels (Stop Loss & Take Profit)
        if (showRiskLevels && riskLevels) {
            const lastHistoricalDate = dates[dates.length - 1];
            const forecastLastDate = predictions && predictions.length > 0
                ? predictions[predictions.length - 1].date
                : dates[dates.length - 1];

            const riskDates = [lastHistoricalDate, forecastLastDate];

            // Stop Loss
            trs.push({
                x: riskDates,
                y: [riskLevels.stop_loss, riskLevels.stop_loss],
                type: 'scatter',
                mode: 'lines',
                name: `Stop Loss ($${riskLevels.stop_loss})`,
                line: { color: '#ef4444', width: 2, dash: 'dash' }
            });

            // Take Profit 1
            trs.push({
                x: riskDates,
                y: [riskLevels.take_profit_1, riskLevels.take_profit_1],
                type: 'scatter',
                mode: 'lines',
                name: `Take Profit 1 ($${riskLevels.take_profit_1})`,
                line: { color: '#10b981', width: 2, dash: 'dot' }
            });

            // Take Profit 2
            trs.push({
                x: riskDates,
                y: [riskLevels.take_profit_2, riskLevels.take_profit_2],
                type: 'scatter',
                mode: 'lines',
                name: `Take Profit 2 ($${riskLevels.take_profit_2})`,
                line: { color: '#059669', width: 1.5, dash: 'dot' }
            });
        }

        // Fibonacci Retracements
        if (showFibonacci && riskLevels?.fibonacci) {
            const fib = riskLevels.fibonacci;
            const xSpan = [dates[Math.max(0, dates.length - 30)], dates[dates.length - 1]];

            trs.push({
                x: xSpan,
                y: [fib.fib_236, fib.fib_236],
                type: 'scatter',
                mode: 'lines',
                name: `Fib 23.6% ($${fib.fib_236})`,
                line: { color: '#eab308', width: 1, dash: 'dot' }
            });
            trs.push({
                x: xSpan,
                y: [fib.fib_500, fib.fib_500],
                type: 'scatter',
                mode: 'lines',
                name: `Fib 50.0% ($${fib.fib_500})`,
                line: { color: '#f97316', width: 1, dash: 'dot' }
            });
            trs.push({
                x: xSpan,
                y: [fib.fib_618, fib.fib_618],
                type: 'scatter',
                mode: 'lines',
                name: `Fib 61.8% ($${fib.fib_618})`,
                line: { color: '#ef4444', width: 1, dash: 'dot' }
            });
        }

        return trs;
    }, [dates, close, sma20, sma50, ema9, ema21, upperBB, lowerBB, showSMA, showEMA, showBB, showConfidence, showRiskLevels, showFibonacci, predictions, riskLevels]);

    const layout = useMemo(() => ({
        title: {
            text: `${company || symbol} Price Analytics`,
            font: { color: '#f3f4f6', size: 18 }
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: {
            gridcolor: 'rgba(255, 255, 255, 0.05)',
            color: '#9ca3af',
            rangeslider: { visible: false },
            type: 'category'
        },
        yaxis: {
            gridcolor: 'rgba(255, 255, 255, 0.05)',
            color: '#9ca3af',
            side: 'right',
            tickprefix: '$'
        },
        margin: { l: 20, r: 60, t: 40, b: 40 },
        autosize: true,
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.15,
            font: { color: '#9ca3af', size: 11 }
        },
        hovermode: 'x unified'
    }), [company, symbol]);

    return (
        <div className="card" style={{ padding: '1rem', marginTop: '1rem' }}>
            {/* Interactive Layer Toggles */}
            <div style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '0.5rem',
                marginBottom: '0.75rem',
                paddingBottom: '0.75rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.05)'
            }}>
                <button
                    className={`btn-pill ${showSMA ? 'active-green' : ''}`}
                    onClick={() => setShowSMA(!showSMA)}
                >
                    SMA (20/50)
                </button>
                <button
                    className={`btn-pill ${showEMA ? 'active-purple' : ''}`}
                    onClick={() => setShowEMA(!showEMA)}
                >
                    EMA (9/21)
                </button>
                <button
                    className={`btn-pill ${showBB ? 'active-blue' : ''}`}
                    onClick={() => setShowBB(!showBB)}
                >
                    Bollinger Bands
                </button>
                {predictions && predictions.length > 0 && (
                    <button
                        className={`btn-pill ${showConfidence ? 'active-purple' : ''}`}
                        onClick={() => setShowConfidence(!showConfidence)}
                    >
                        95% Confidence Band
                    </button>
                )}
                {riskLevels && (
                    <button
                        className={`btn-pill ${showRiskLevels ? 'active-teal' : ''}`}
                        onClick={() => setShowRiskLevels(!showRiskLevels)}
                    >
                        ATR Risk Levels
                    </button>
                )}
                {riskLevels?.fibonacci && (
                    <button
                        className={`btn-pill ${showFibonacci ? 'active-blue' : ''}`}
                        onClick={() => setShowFibonacci(!showFibonacci)}
                    >
                        Fibonacci Retracements
                    </button>
                )}
            </div>

            <div style={{ width: '100%', height: '520px' }}>
                <Plot
                    data={traces}
                    layout={layout}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '100%' }}
                    config={{
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['lasso2d', 'select2d']
                    }}
                />
            </div>
        </div>
    );
};

export default React.memo(StockChart);
