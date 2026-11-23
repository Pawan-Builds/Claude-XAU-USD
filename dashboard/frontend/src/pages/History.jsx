import React, { useState, useEffect } from 'react'
import { api } from '../services/api'
import { formatCurrency, formatNumber, formatDateTime, formatDuration } from '../utils/format'
import { Search, Filter, Download } from 'lucide-react'

export default function History() {
  const [trades, setTrades] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filters, setFilters] = useState({
    search: '',
    direction: '',
    strategy: '',
    exitReason: '',
    dateFrom: '',
    dateTo: ''
  })
  const [sortConfig, setSortConfig] = useState({ key: 'exit_time', direction: 'desc' })
  const [pagination, setPagination] = useState({ page: 1, pageSize: 50 })
  const [strategies, setStrategies] = useState([])
  const [exitReasons, setExitReasons] = useState([])

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await api.get('/api/history', { params: { limit: 500 } })
        setTrades(res.data.trades)
        setError(null)

        // Extract unique strategies and exit reasons
        const uniqueStrategies = [...new Set(res.data.trades.map(t => t.strategy).filter(Boolean))]
        const uniqueExitReasons = [...new Set(res.data.trades.map(t => t.exit_reason).filter(Boolean))]
        setStrategies(uniqueStrategies)
        setExitReasons(uniqueExitReasons)
      } catch (err) {
        setError('Failed to load trade history')
      } finally {
        setLoading(false)
      }
    }

    fetchHistory()
  }, [])

  // Filter and sort trades
  const filteredTrades = trades.filter(trade => {
    if (filters.search) {
      const search = filters.search.toLowerCase()
      if (!trade.direction.toLowerCase().includes(search) &&
          !trade.strategy?.toLowerCase().includes(search) &&
          !trade.ticket.toString().includes(search)) {
        return false
      }
    }
    if (filters.direction && trade.direction !== filters.direction) return false
    if (filters.strategy && trade.strategy !== filters.strategy) return false
    if (filters.exitReason && trade.exit_reason !== filters.exitReason) return false
    if (filters.dateFrom && trade.exit_time < filters.dateFrom) return false
    if (filters.dateTo && trade.exit_time > filters.dateTo) return false
    return true
  }).sort((a, b) => {
    const aVal = a[sortConfig.key]
    const bVal = b[sortConfig.key]
    if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1
    if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1
    return 0
  })

  // Pagination
  const totalPages = Math.ceil(filteredTrades.length / pagination.pageSize)
  const paginatedTrades = filteredTrades.slice(
    (pagination.page - 1) * pagination.pageSize,
    pagination.page * pagination.pageSize
  )

  const handleSort = (key) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
    }))
  }

  const exportCSV = () => {
    const headers = [
      'Ticket', 'Direction', 'Entry Time', 'Exit Time',
      'Entry Price', 'Exit Price', 'SL', 'TP', 'Volume',
      'Gross Profit', 'Net Profit', 'Realized R', 'Risk %',
      'Strategy', 'Exit Reason', 'Duration (hours)'
    ]

    const rows = filteredTrades.map(t => [
      t.ticket, t.direction, formatDateTime(t.entry_time), formatDateTime(t.exit_time),
      t.entry_price, t.exit_price, t.sl, t.tp, t.volume,
      t.gross_profit, t.net_profit, t.realized_r, t.risk_percent,
      t.strategy, t.exit_reason, t.duration_hours
    ])

    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `trades_${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (loading) return <div className="loading">Loading history...</div>
  if (error) return <div className="error">{error}</div>

  return (
    <div className="page">
      <div className="page-header">
        <h1>Trade History</h1>
        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn btn-secondary" onClick={exportCSV}>
            <Download size={16} /> Export CSV
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="card-header">
          <span className="card-title">Filters</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          <div>
            <label className="stat-label">Search</label>
            <input
              className="input"
              placeholder="Ticket, direction, strategy..."
              value={filters.search}
              onChange={e => setFilters(prev => ({ ...prev, search: e.target.value, page: 1 }))}
            />
          </div>
          <div>
            <label className="stat-label">Direction</label>
            <select className="input" value={filters.direction} onChange={e => setFilters(prev => ({ ...prev, direction: e.target.value, page: 1 }))}>
              <option value="">All</option>
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
            </select>
          </div>
          <div>
            <label className="stat-label">Strategy</label>
            <select className="input" value={filters.strategy} onChange={e => setFilters(prev => ({ ...prev, strategy: e.target.value, page: 1 }))}>
              <option value="">All</option>
              {strategies.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <label className="stat-label">Exit Reason</label>
            <select className="input" value={filters.exitReason} onChange={e => setFilters(prev => ({ ...prev, exitReason: e.target.value, page: 1 }))}>
              <option value="">All</option>
              {exitReasons.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>
          <div>
            <label className="stat-label">Date From</label>
            <input className="input" type="date" value={filters.dateFrom} onChange={e => setFilters(prev => ({ ...prev, dateFrom: e.target.value, page: 1 }))} />
          </div>
          <div>
            <label className="stat-label">Date To</label>
            <input className="input" type="date" value={filters.dateTo} onChange={e => setFilters(prev => ({ ...prev, dateTo: e.target.value, page: 1 }))} />
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <span className="stat-label">Showing {paginatedTrades.length} of {filteredTrades.length} trades</span>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                {[
                  { key: 'ticket', label: 'Ticket' },
                  { key: 'direction', label: 'Dir' },
                  { key: 'entry_time', label: 'Entry Time' },
                  { key: 'exit_time', label: 'Exit Time' },
                  { key: 'entry_price', label: 'Entry' },
                  { key: 'exit_price', label: 'Exit' },
                  { key: 'sl', label: 'SL' },
                  { key: 'tp', label: 'TP' },
                  { key: 'volume', label: 'Vol' },
                  { key: 'net_profit', label: 'P/L' },
                  { key: 'realized_r', label: 'R' },
                  { key: 'risk_percent', label: 'Risk%' },
                  { key: 'strategy', label: 'Strategy' },
                  { key: 'exit_reason', label: 'Exit' },
                  { key: 'duration_hours', label: 'Dur.' }
                ].map(col => (
                  <th key={col.key} onClick={() => handleSort(col.key)} style={{ cursor: 'pointer', userSelect: 'none' }}>
                    {col.label}
                    {sortConfig.key === col.key && (
                      <span>{sortConfig.direction === 'asc' ? ' ▲' : ' ▼'}</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {paginatedTrades.map(trade => (
                <tr key={trade.ticket}>
                  <td className="mono">{trade.ticket}</td>
                  <td><span className={`badge ${trade.direction === 'BUY' ? 'badge-green' : 'badge-red'}`}>{trade.direction}</span></td>
                  <td>{formatDateTime(trade.entry_time)}</td>
                  <td>{formatDateTime(trade.exit_time)}</td>
                  <td className="mono">{trade.entry_price.toFixed(2)}</td>
                  <td className="mono">{trade.exit_price.toFixed(2)}</td>
                  <td className="mono">{trade.sl?.toFixed(2) || 'N/A'}</td>
                  <td className="mono">{trade.tp?.toFixed(2) || 'N/A'}</td>
                  <td className="mono">{trade.volume}</td>
                  <td className="mono" style={{ color: trade.net_profit >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                    {trade.net_profit >= 0 ? '+' : ''}{formatCurrency(trade.net_profit)}
                  </td>
                  <td className="mono" style={{ color: trade.realized_r >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                    {trade.realized_r >= 0 ? '+' : ''}{trade.realized_r.toFixed(2)}R
                  </td>
                  <td className="mono">{trade.risk_percent}%</td>
                  <td><span className="badge badge-blue">{trade.strategy || 'N/A'}</span></td>
                  <td><span className={`badge ${trade.exit_reason === 'TP' ? 'badge-green' : trade.exit_reason === 'SL' ? 'badge-red' : 'badge-gray'}`}>{trade.exit_reason}</span></td>
                  <td className="mono">{formatDuration(trade.duration_hours)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', marginTop: '16px' }}>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => setPagination(prev => ({ ...prev, page: prev.page - 1 }))}
              disabled={pagination.page === 1}
            >
              Previous
            </button>
            <span style={{ display: 'flex', alignItems: 'center', color: 'var(--text-secondary)' }}>
              Page {pagination.page} of {totalPages}
            </span>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => setPagination(prev => ({ ...prev, page: prev.page + 1 }))}
              disabled={pagination.page === totalPages}
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  )
}