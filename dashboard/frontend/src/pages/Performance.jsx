import React, { useState, useEffect } from 'react'
import { api } from '../services/api'
import Plot from 'react-plotly.js'
import { formatCurrency, formatNumber, formatPercent } from '../utils/format'
import { BarChart, TrendingUp, TrendingDown, Target, DollarSign } from 'lucide-react'

export default function Performance() {
  const [performance, setPerformance] = useState(null)
  const [equityCurve, setEquityCurve] = useState([])
  const [monthlyReturns, setMonthlyReturns] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchPerformance = async () => {
      try {
        const res = await api.get('/api/performance')
        setPerformance(res.data)
        setEquityCurve(res.data.equity_curve || [])
        setMonthlyReturns(res.data.monthly_returns || {})
        setError(null)
      } catch (err) {
        setError('Failed to load performance data')
      } finally {
        setLoading(false)
      }
    }

    fetchPerformance()
  }, [])

  if (loading) return <div className="loading">Loading performance...</div>
  if (error) return <div className="error">{error}</div>
  if (!performance || performance.trades === 0) {
    return (
      <div className="page">
        <div className="card empty-state">
          <TrendingUp size={48} style={{ color: 'var(--text-muted)', marginBottom: '16px' }} />
          <h3>No Performance Data</h3>
          <p>Trade history will appear here after trades are closed.</p>
        </div>
      </div>
    )
  }

  // Equity curve chart
  const equityData = [{
    x: equityCurve.map(p => p.date),
    y: equityCurve.map(p => p.equity),
    type: 'scatter',
    mode: 'lines',
    name: 'Equity',
    line: { color: '#58a6ff', width: 2 },
    fill: 'tozeroy',
    fillcolor: 'rgba(88, 166, 255, 0.1)'
  }]

  const equityLayout = {
    paper_bgcolor: 'var(--bg-secondary)',
    plot_bgcolor: 'var(--bg-primary)',
    font: { color: 'var(--text-primary)', family: 'Inter' },
    margin: { t: 10, r: 10, b: 40, l: 60 },
    xaxis: { gridcolor: 'var(--border-color)', title: '' },
    yaxis: { gridcolor: 'var(--border-color)', title: 'Equity ($)' },
    showlegend: false,
    hovermode: 'x unified'
  }

  // Monthly returns chart
  const months = Object.keys(monthlyReturns).sort()
  const monthlyData = [{
    x: months,
    y: months.map(m => monthlyReturns[m]),
    type: 'bar',
    name: 'Monthly Return %',
    marker: {
      color: months.map(m => monthlyReturns[m] >= 0 ? 'var(--accent-green)' : 'var(--accent-red)')
    }
  }]

  const monthlyLayout = {
    paper_bgcolor: 'var(--bg-secondary)',
    plot_bgcolor: 'var(--bg-primary)',
    font: { color: 'var(--text-primary)', family: 'Inter' },
    margin: { t: 10, r: 10, b: 40, l: 60 },
    xaxis: { gridcolor: 'var(--border-color)', title: '' },
    yaxis: { gridcolor: 'var(--border-color)', title: 'Return (%)' },
    showlegend: false
  }

  // Win/Loss distribution
  const winLossData = [{
    labels: ['Wins', 'Losses'],
    values: [performance.winning_trades, performance.losing_trades],
    type: 'pie',
    hole: 0.5,
    marker: {
      colors: ['var(--accent-green)', 'var(--accent-red)']
    },
    textinfo: 'label+percent',
    textfont: { color: 'var(--text-primary)', size: 14 }
  }]

  const pieLayout = {
    paper_bgcolor: 'var(--bg-secondary)',
    plot_bgcolor: 'var(--bg-primary)',
    font: { color: 'var(--text-primary)', family: 'Inter' },
    margin: { t: 10, r: 10, b: 10, l: 10 },
    showlegend: false
  }

  const stats = [
    { label: 'Total Trades', value: performance.total_trades, icon: 'list' },
    { label: 'Win Rate', value: `${performance.win_rate}%`, icon: 'target', positive: performance.win_rate >= 50 },
    { label: 'Profit Factor', value: typeof performance.profit_factor === 'number' ? performance.profit_factor.toFixed(2) : performance.profit_factor, icon: 'ratio', positive: typeof performance.profit_factor === 'number' && performance.profit_factor > 1 },
    { label: 'Net Profit', value: formatCurrency(performance.net_profit), icon: 'dollar-sign', trendPositive: performance.net_profit >= 0 },
    { label: 'Gross Profit', value: formatCurrency(performance.gross_profit), icon: 'trending-up' },
    { label: 'Gross Loss', value: formatCurrency(performance.gross_loss), icon: 'trending-down' },
    { label: 'Avg Win', value: formatCurrency(performance.avg_win), icon: 'trending-up' },
    { label: 'Avg Loss', value: formatCurrency(performance.avg_loss), icon: 'trending-down' },
    { label: 'Expectancy', value: formatCurrency(performance.expectancy), icon: 'target', positive: performance.expectancy > 0 },
  ]

  return (
    <div className="page">
      <div className="page-header">
        <h1>Performance Analytics</h1>
      </div>

      {/* Key Stats */}
      <div className="grid grid-5">
        {stats.map((stat, i) => (
          <div key={i} className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <span className="stat-icon" data-icon={stat.icon} />
              <span className="stat-label">{stat.label}</span>
            </div>
            <div className="stat-value" style={{ color: stat.positive === false ? 'var(--accent-red)' : stat.positive || stat.trendPositive ? 'var(--accent-green)' : 'inherit' }}>
              {stat.value}
            </div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-2" style={{ marginTop: '20px' }}>
        <div className="card" style={{ height: '400px' }}>
          <div className="card-header">
            <span className="card-title">Equity Curve</span>
          </div>
          <div style={{ height: '340px' }}>
            <Plot data={equityData} layout={equityLayout} config={{ displayModeBar: false }} />
          </div>
        </div>

        <div className="card" style={{ height: '400px' }}>
          <div className="card-header">
            <span className="card-title">Monthly Returns</span>
          </div>
          <div style={{ height: '340px' }}>
            <Plot data={monthlyData} layout={monthlyLayout} config={{ displayModeBar: false }} />
          </div>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: '20px' }}>
        <div className="card" style={{ height: '350px' }}>
          <div className="card-header">
            <span className="card-title">Win / Loss Distribution</span>
          </div>
          <div style={{ height: '290px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Plot data={winLossData} layout={pieLayout} config={{ displayModeBar: false }} />
          </div>
        </div>

        <div className="card" style={{ height: '350px' }}>
          <div className="card-header">
            <span className="card-title">Additional Metrics</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', padding: '8px' }}>
            <div style={{ padding: '16px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
              <div className="stat-label">Sharpe Ratio</div>
              <div className="stat-value" style={{ fontSize: '28px' }}>{performance.sharpe_ratio ? performance.sharpe_ratio.toFixed(2) : 'N/A'}</div>
            </div>
            <div style={{ padding: '16px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
              <div className="stat-label">Sortino Ratio</div>
              <div className="stat-value" style={{ fontSize: '28px' }}>{performance.sortino_ratio ? performance.sortino_ratio.toFixed(2) : 'N/A'}</div>
            </div>
            <div style={{ padding: '16px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
              <div className="stat-label">Recovery Factor</div>
              <div className="stat-value" style={{ fontSize: '28px' }}>{performance.recovery_factor ? performance.recovery_factor.toFixed(2) : 'N/A'}</div>
            </div>
            <div style={{ padding: '16px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
              <div className="stat-label">Max Drawdown</div>
              <div className="stat-value negative" style={{ fontSize: '28px' }}>{performance.max_drawdown_pct ? performance.max_drawdown_pct.toFixed(2) + '%' : 'N/A'}</div>
            </div>
            <div style={{ padding: '16px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
              <div className="stat-label">Largest Win</div>
              <div className="stat-value positive" style={{ fontSize: '28px' }}>{formatCurrency(performance.largest_win)}</div>
            </div>
            <div style={{ padding: '16px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
              <div className="stat-label">Largest Loss</div>
              <div className="stat-value negative" style={{ fontSize: '28px' }}>{formatCurrency(performance.largest_loss)}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}