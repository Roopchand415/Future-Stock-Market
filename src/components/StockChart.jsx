import React, { useState } from 'react';
import Plot from 'react-plotly.js';

const StockChart = ({ data, predictions, symbol, company, riskLevels }) => {
    if (!data || data.length === 0) return null;

    const [showSMA, setShowSMA] = useState(true);
    const [showEMA, setShowEMA] = useState(false);
    const [showBB, setShowBB] = useState(false);
    const [showConfidence, setShowConfidence] = useState(true);
    const [showRiskLevels, setShowRiskLevels] = useState(true);

    const dates = data.map(d => d.Date);
    const close = data.map(d => d.Close);
    const sma20 = data.map(d => d.SMA_20);
    const sma50 = data.map(d => d.SMA_50);
    const ema9 = data.map(d => d.EMA_9);
    const ema21 = data.map(d => d.EMA_21);
    const upperBB = data.map(d => d.Upper_Band);
    const lowerBB = data.map(d => d.Lower_Band);

    const traces = [
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
            traces.push({
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
            traces.push({
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
            traces.push({
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
            traces.push({
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
        traces.push({
            x: dates,
            y: upperBB,
            type: 'scatter',
            mode: 'lines',
            name: 'Upper BB',
            line: { color: 'rgba(156, 163, 175, 0.4)', width: 1 },
            showlegend: false
        });
        traces.push({
            x: dates,
            y: lowerBB,
            type: 'scatter',
            mode: 'lines',
            name: 'Bollinger Bands',
            fill: 'tonexty',
            fillcolor: 'rgba(156, 163, 175, 0.08)',
            line: { color: 'rgba(156, 163, 175, 0.4)', width: 1 },
            opacity: 0.5
        });
    }

    // Predictions & Probabilistic Confidence Bands
    if (predictions && predictions.length > 0) {
        const predDates = predictions.map(p => p.date);
        const predPrices = predictions.map(p => p.price);
        const upper95 = predictions.map(p => p.upper_95 || p.price * 1.03);
        const lower95 = predictions.map(p => p.lower_95 || p.price * 0.97);

        const lastDataDate = dates[dates.length - 1];
        const lastDataPrice = close[close.length - 1];

        // Connection line
        traces.push({
            x: [lastDataDate, predDates[0]],
            y: [lastDataPrice, predPrices[0]],
            mode: 'lines',
            line: { color: '#ef4444', dash: 'dot', width: 2 },
            showlegend: false,
            hoverinfo: 'skip'
        });

        // Shaded Confidence Corridor (95% Probability Bounds)
        if (showConfidence && upper95.length > 0 && lower95.length > 0) {
            const corridorX = [lastDataDate, ...predDates];
            const corridorUpper = [lastDataPrice, ...upper95];
            const corridorLower = [lastDataPrice, ...lower95];

            traces.push({
                x: corridorX,
                y: corridorUpper,
                type: 'scatter',
                mode: 'lines',
                name: '95% Confidence Upper',
                line: { color: 'rgba(239, 68, 68, 0.2)', width: 1 },
                showlegend: false
            });
            traces.push({
                x: corridorX,
                y: corridorLower,
                type: 'scatter',
                mode: 'lines',
                name: '95% Confidence Band',
                fill: 'tonexty',
                fillcolor: 'rgba(239, 68, 68, 0.12)',
                line: { color: 'rgba(239, 68, 68, 0.2)', width: 1 },
                opacity: 0.5
            });
        }

        // Prediction Trend Line
        traces.push({
            x: predDates,
            y: predPrices,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'AI Forecast',
            line: { color: '#ef4444', dash: 'dash', width: 2.5 },
            marker: { size: 6, color: '#ef4444' }
        });
    }

    // Risk Management Horizontal Targets (Stop Loss / Take Profits)
    const shapes = [];
    const annotations = [];

    if (showRiskLevels && riskLevels) {
        const lastDate = dates[dates.length - 1];
        const firstDate = dates[Math.max(0, dates.length - 30)];

        if (riskLevels.stop_loss) {
            shapes.push({
                type: 'line',
                x0: firstDate,
                x1: lastDate,
                y0: riskLevels.stop_loss,
                y1: riskLevels.stop_loss,
                line: { color: '#ef4444', width: 1.5, dash: 'dash' }
            });
            annotations.push({
                x: lastDate,
                y: riskLevels.stop_loss,
                text: `SL: $${riskLevels.stop_loss}`,
                showarrow: false,
                font: { color: '#ef4444', size: 10 },
                xanchor: 'left'
            });
        }

        if (riskLevels.take_profit_2) {
            shapes.push({
                type: 'line',
                x0: firstDate,
                x1: lastDate,
                y0: riskLevels.take_profit_2,
                y1: riskLevels.take_profit_2,
                line: { color: '#10b981', width: 1.5, dash: 'dash' }
            });
            annotations.push({
                x: lastDate,
                y: riskLevels.take_profit_2,
                text: `TP2: $${riskLevels.take_profit_2}`,
                showarrow: false,
                font: { color: '#10b981', size: 10 },
                xanchor: 'left'
            });
        }
    }

    const chartTitle = company && company !== symbol ? `${symbol} - ${company}` : symbol;

    return (
        <div className="card" style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
            {/* Chart Toolbar Controls */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', paddingBottom: '0.4rem', borderBottom: '1px solid rgba(255, 255, 255, 0.08)' }}>
                <span style={{ fontWeight: '700', fontSize: '1rem', color: 'var(--text-primary)' }}>
                    {chartTitle}
                </span>

                <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                    <button
                        onClick={() => setShowSMA(!showSMA)}
                        className={`btn-pill ${showSMA ? 'active-blue' : ''}`}
                        title="Simple Moving Averages (20 / 50)"
                    >
                        SMA (20/50)
                    </button>
                    <button
                        onClick={() => setShowEMA(!showEMA)}
                        className={`btn-pill ${showEMA ? 'active-purple' : ''}`}
                        title="Exponential Moving Averages (9 / 21)"
                    >
                        EMA (9/21)
                    </button>
                    <button
                        onClick={() => setShowBB(!showBB)}
                        className={`btn-pill ${showBB ? 'active-teal' : ''}`}
                        title="Bollinger Bands"
                    >
                        Bollinger Bands
                    </button>
                    {predictions && (
                        <button
                            onClick={() => setShowConfidence(!showConfidence)}
                            className={`btn-pill ${showConfidence ? 'active-red' : ''}`}
                            title="95% Forecast Confidence Corridor"
                        >
                            Confidence Corridor
                        </button>
                    )}
                    {riskLevels && (
                        <button
                            onClick={() => setShowRiskLevels(!showRiskLevels)}
                            className={`btn-pill ${showRiskLevels ? 'active-green' : ''}`}
                            title="Stop Loss and Take Profit lines"
                        >
                            Risk Levels
                        </button>
                    )}
                </div>
            </div>

            {/* Plotly Canvas */}
            <div style={{ height: '480px', width: '100%' }}>
                <Plot
                    data={traces}
                    layout={{
                        autosize: true,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#f3f4f6', family: 'Inter, sans-serif' },
                        xaxis: {
                            gridcolor: '#374151',
                            showgrid: true,
                            zerolinecolor: '#374151'
                        },
                        yaxis: {
                            gridcolor: '#374151',
                            showgrid: true,
                            zerolinecolor: '#374151'
                        },
                        margin: { t: 20, r: 40, l: 50, b: 40 },
                        legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' },
                        hovermode: 'x unified',
                        shapes: shapes,
                        annotations: annotations
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler={true}
                />
            </div>
        </div>
    );
};

export default StockChart;
