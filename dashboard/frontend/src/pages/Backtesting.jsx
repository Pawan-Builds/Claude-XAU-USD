import React, { useState, useEffect } from 'react'
import { api } from '../services/api'
import { formatCurrency, formatNumber, formatPercent, formatDateTime } from '../utils/format'
import { Play, FlaskConical, Download, FileText, Trash2, ChevronLeft, ChevronRight } from 'lucide-react'

export default function Backtesting() {
  const [reports, setReports] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showConfig, setShowConfig] = useState(false)
  const [runningBacktest, setRunningBacktest] = useState(null)
  const [selectedReport, setSelectedReport] = useState(null)

  // Config form state
  const [config, setConfig] = useState({
    name: `Backtest_${new Date().toISOString().split('T')[0]}`,
    start_date: '2023-01-01',
    end_date: '2024-12-31',
    initial_balance: 10000,
    risk_percent: 2.0,
    min_risk_reward: 2.0,
    timeframes: ['m5', 'm15', 'h1', 'h4'],
    sessions: ['london', 'new_york'],
    strategies: ['liquidity_sweep_fvg'],
    spread_pips: 2.0,
    slippage_pips: 1.0,
    commission_per_lot: 7.0,
    swap_enabled: true
  })

  const [timeframeOptions] = useState([
    { value: 'm5', label: 'M5' },
    { value: 'm15', label: 'M15' },
    { value: 'h1', label: 'H1' },
    { value: 'h4', label: 'H4' }
  ])

  const [sessionOptions] = useState([
    { value: 'london', label: 'London' },
    { value: 'new_york', label: 'New York' },
    { value: 'asian', label: 'Asian' }
  ])

  const [strategyOptions] = useState([
    { value: 'liquidity_sweep_fvg', label: 'Liquidity Sweep + FVG' },
    { value: 'liquidity_sweep_choch', label: 'Liquidity Sweep + CHoCH' },
    { value: 'bos_order_block', label: 'BOS + Order Block' },
    { value: 'choch_order_block', label: 'CHoCH + Order Block' },
    { value: 'displacement_fvg', label: 'Displacement + FVG' },
    { value: 'liquidity_sweep_ob_fvg', label: 'Liquidity Sweep + OB + FVG' }
  ])

  useEffect(() => {
    fetchReports()
  }, [])

  const fetchReports = async () => {
    try {
      const res = await api.get('/api/backtests')
      setReports(res.data.reports)
      setError(null)
    } catch (err) {
      setError('Failed to load backtest reports')
    } finally {
      setLoading(false)
    }
  }

  const handleConfigChange = (field, value) => {
    setConfig(prev => ({ ...prev, [field]: value }))
  }

  const handleMultiSelect = (field, value) => {
    setConfig(prev => {
      const current = prev[field] || []
      const updated = current.includes(value)
        ? current.filter(v => v !== value)
        : [...current, value]
      return { ...prev, [field]: updated }
    })
  }

  const runBacktest = async () => {
    if (!config.name || !config.start_date || !config.end_date) {
      alert('Please fill in all required fields')
      return
    }

    try {
      const res = await api.post('/api/backtests', config)
      setRunningBacktest(res.data.config_id)
      setShowConfig(false)
      // Poll for completion
      pollBacktest(res.data.config_id)
    } catch (err) {
      alert('Failed to start backtest: ' + err.message)
    }
  }

  const pollBacktest = async (configId) => {
    const checkStatus = async () => {
      try {
        // In a real implementation, you'd have a status endpoint
        // For now, just refresh reports periodically
        await fetchReports()
      } catch (err) {
        console.error('Poll error:', err)
      }
    }

    const interval = setInterval(checkStatus, 5000)
    // Stop polling after 10 minutes
    setTimeout(() => clearInterval(interval), 600000)
  }

  const viewReport = (report) => {
    setSelectedReport(report)
  }

  const exportReport = async (reportId) => {
    try {
      // In a real implementation, you'd call an export endpoint
      alert('Export functionality would download the report')
    } catch (err) {
      alert('Export failed: ' + err.message)
    }
  }

  const deleteReport = async (reportId) => {
    if (!window.confirm('Delete this backtest report?')) return
    // In a real implementation, you'd call a delete endpoint
    alert('Delete functionality not implemented')
  }

  if (loading) return <div className="loading">Loading backtests...</div>

  return (
    <div className="page">
      <div className="page-header">
        <h1>Backtesting</h1>
        <button className="btn btn-primary" onClick={() => setShowConfig(true)}>
          <Play size={16} /> New Backtest
        </button>
      </div>

      {/* Backtest Config Modal */}
      {showConfig && (
        <div className="modal-overlay" onClick={() => setShowConfig(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>New Backtest Configuration</h2>
              <button className="modal-close" onClick={() => setShowConfig(false)}>×</button>
            </div>
            <div className="modal-body">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <label className="stat-label">Name</label>
                  <input className="input" value={config.name} onChange={e => handleConfigChange('name', e.target.value)} />
                </div>
                <div>
                  <label className="stat-label">Initial Balance</label>
                  <input className="input" type="number" value={config.initial_balance} onChange={e => handleConfigChange('initial_balance', parseFloat(e.target.value))} />
                </div>
                <div>
                  <label className="stat-label">Start Date</label>
                  <input className="input" type="date" value={config.start_date} onChange={e => handleConfigChange('start_date', e.target.value)} />
                </div>
                <div>
                  <label className="stat-label">End Date</label>
                  <input className="input" type="date" value={config.end_date} onChange={e => handleConfigChange('end_date', e.target.value)} />
                </div>
                <div>
                  <label className="stat-label">Risk %</label>
                  <input className="input" type="number" step="0.1" value={config.risk_percent} onChange={e => handleConfigChange('risk_percent', parseFloat(e.target.value))} />
                </div>
                <div>
                  <label className="stat-label">Min R:R</label>
                  <input className="input" type="number" step="0.1" value={config.min_risk_reward} onChange={e => handleConfigChange('min_risk_reward', parseFloat(e.target.value))} />
                </div>
                <div>
                  <label className="stat-label">Spread (pips)</label>
                  <input className="input" type="number" step="0.1" value={config.spread_pips} onChange={e => handleConfigChange('spread_pips', parseFloat(e.target.value))} />
                </div>
                <div>
                  <label className="stat-label">Slippage (pips)</label>
                  <input className="input" type="number" step="0.1" value={config.slippage_pips} onChange={e => handleConfigChange('slippage_pips', parseFloat(e.target.value))} />
                </div>
                <div>
                  <label className="stat-label">Commission ($/lot)</label>
                  <input className="input" type="number" step="0.1" value={config.commission_per_lot} onChange={e => handleConfigChange('commission_per_lot', parseFloat(e.target.value))} />
                </div>
              </div>

              <div style={{ marginTop: '20px' }}>
                <label className="stat-label">Timeframes</label>
                <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginTop: '8px' }}>
                  {timeframeOptions.map(tf => (
                    <label key={tf.value} style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={config.timeframes.includes(tf.value)}
                        onChange={e => handleMultiSelect('timeframes', tf.value)}
                      />
                      <span>{tf.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div style={{ marginTop: '16px' }}>
                <label className="stat-label">Sessions</label>
                <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginTop: '8px' }}>
                  {sessionOptions.map(s => (
                    <label key={s.value} style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={config.sessions.includes(s.value)}
                        onChange={e => handleMultiSelect('sessions', s.value)}
                      />
                      <span>{s.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div style={{ marginTop: '16px' }}>
                <label className="stat-label">Strategies</label>
                <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginTop: '8px' }}>
                  {strategyOptions.map(s => (
                    <label key={s.value} style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={config.strategies.includes(s.value)}
                        onChange={e => handleMultiSelect('strategies', s.value)}
                      />
                      <span>{s.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div style={{ marginTop: '16px' }}>
                <label className="stat-label" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="checkbox"
                    checked={config.swap_enabled}
                    onChange={e => handleConfigChange('swap_enabled', e.target.checked)}
                  />
                  Enable Swaps
                </label>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setShowConfig(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={runBacktest} disabled={runningBacktest}>
                <Play size={16} /> {runningBacktest ? 'Running...' : 'Run Backtest'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Report Detail Modal */}
      {selectedReport && (
        <div className="modal-overlay" onClick={() => setSelectedReport(null)}>
          <div className="modal modal-large" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>{selectedReport.config?.name || 'Backtest Report'}</h2>
              <button className="modal-close" onClick={() => setSelectedReport(null)}>×</button>
            </div>
            <div className="modal-body" style={{ maxHeight: '70vh', overflow: 'auto' }}>
              {selectedReport.summary && (
                <div style={{ marginBottom: '20px' }}>
                  <h3>Summary</h3>
                  <div className="grid grid-4">
                    <div className="card"><div className="stat-label">Net Profit</div><div className="stat-value">{formatCurrency(selectedReport.summary.net_profit)}</div></div>
                    <div className="card"><div className="stat-label">Return</div><div className="stat-value">{formatPercent(selectedReport.summary.net_return_pct)}</div></div>
                    <div className="card"><div className="stat-label">Trades</div><div className="stat-value">{selectedReport.summary.total_trades}</div></div>
                    <div className="card"><div className="stat-label">Win Rate</div><div className="stat-value">{formatPercent(selectedReport.summary.win_rate)}</div></div>
                    <div className="card"><div className="stat-label">Profit Factor</div><div className="stat-value">{typeof selectedReport.summary.profit_factor === 'number' ? selectedReport.summary.profit_factor.toFixed(2) : selectedReport.summary.profit_factor}</div></div>
                    <div className="card"><div className="stat-label">Max DD</div><div className="stat-value negative">{formatPercent(selectedReport.summary.max_drawdown_pct)}</div></div>
                    <div className="card"><div className="stat-label">Sharpe</div><div className="stat-value">{selectedReport.summary.sharpe_ratio?.toFixed(2)}</div></div>
                    <div className="card"><div className="stat-label">Expectancy</div><div className="stat-value">{formatCurrency(selectedReport.summary.expectancy)}</div></div>
                  </div>
                </div>
              )}

              {selectedReport.trades && selectedReport.trades.length > 0 && (
                <div>
                  <h3>Trades ({selectedReport.trades.length})</h3>
                  <div className="table-container">
                    <table>
                      <thead>
                        <tr>
                          <th>Ticket</th>
                          <th>Dir</th>
                          <th>Entry</th>
                          <th>Exit</th>
                          <th>P/L</th>
                          <th>R</th>
                          <th>Strategy</th>
                          <th>Exit</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedReport.trades.slice(0, 50).map(t => (
                          <tr key={t.ticket}>
                            <td className="mono">{t.ticket}</td>
                            <td><span className={`badge ${t.direction === 'BUY' ? 'badge-green' : 'badge-red'}`}>{t.direction}</span></td>
                            <td className="mono">{t.entry_price.toFixed(2)}</td>
                            <td className="mono">{t.exit_price.toFixed(2)}</td>
                            <td className="mono" style={{ color: t.net_profit >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                              {t.net_profit >= 0 ? '+' : ''}{formatCurrency(t.net_profit)}
                            </td>
                            <td className="mono" style={{ color: t.realized_r >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                              {t.realized_r >= 0 ? '+' : ''}{t.realized_r.toFixed(2)}R
                            </td>
                            <td><span className="badge badge-blue">{t.strategy}</span></td>
                            <td><span className={`badge ${t.exit_reason === 'TP' ? 'badge-green' : t.exit_reason === 'SL' ? 'badge-red' : 'badge-gray'}`}>{t.exit_reason}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {selectedReport.trades.length > 50 && <p style={{ color: 'var(--text-secondary)', marginTop: '8px' }}>Showing first 50 of {selectedReport.trades.length} trades</p>}
                </div>
              )}

              <div style={{ display: 'flex', gap: '12px', marginTop: '20px', justifyContent: 'flex-end' }}>
                <button className="btn btn-secondary" onClick={() => exportReport(selectedReport.id)}>
                  <Download size={16} /> Export JSON
                </button>
                <button className="btn btn-secondary" onClick={() => exportReport(selectedReport.id)}>
                  <FileText size={16} /> Export CSV
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Reports List */}
      <div className="card">
        {reports.length === 0 ? (
          <div className="empty-state" style={{ padding: '60px 20px' }}>
            <FlaskConical size={48} style={{ color: 'var(--text-muted)', marginBottom: '16px' }} />
            <h3>No Backtest Reports</h3>
            <p>Run your first backtest to see results here.</p>
          </div>
        ) : (
          <>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Period</th>
                    <th>Initial</th>
                    <th>Final</th>
                    <th>Net P/L</th>
                    <th>Return</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Profit Factor</th>
                    <th>Max DD</th>
                    <th>Completed</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {reports.map(report => (
                    <tr key={report.id}>
                      <td style={{ fontWeight: 600 }}>{report.config?.name || `Report #${report.id}`}</td>
                      <td className="mono">{report.start_date?.split('T')[0]} → {report.end_date?.split('T')[0]}</td>
                      <td className="mono">{formatCurrency(report.initial_capital)}</td>
                      <td className="mono">{formatCurrency(report.final_capital)}</td>
                      <td className="mono" style={{ color: report.net_profit >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                        {report.net_profit >= 0 ? '+' : ''}{formatCurrency(report.net_profit)}
                      </td>
                      <td className="mono" style={{ color: report.net_return_pct >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                        {report.net_return_pct >= 0 ? '+' : ''}{formatPercent(report.net_return_pct)}
                      </td>
                      <td className="mono">{report.total_trades}</td>
                      <td className="mono">{formatPercent(report.win_rate)}</td>
                      <td className="mono">{typeof report.profit_factor === 'number' ? report.profit_factor.toFixed(2) : report.profit_factor}</td>
                      <td className="mono negative">{formatPercent(report.max_drawdown_pct)}</td>
                      <td className="mono">{formatDateTime(report.completed_at)}</td>
                      <td>
                        <button className="btn btn-secondary btn-sm" onClick={() => viewReport(report)}>
                          <FileText size={14} /> View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  )
}