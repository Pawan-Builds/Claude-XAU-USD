import React, { useState, useEffect } from 'react'
import { useLiveUpdate } from '../hooks/useLiveUpdate'
import { api } from '../services/api'
import { formatCurrency, formatNumber, formatDateTime, formatDuration } from '../utils/format'
import { X, AlertTriangle } from 'lucide-react'

export default function Positions() {
  const [positions, setPositions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [closingTicket, setClosingTicket] = useState(null)

  const liveData = useLiveUpdate()

  useEffect(() => {
    const fetchPositions = async () => {
      try {
        const res = await api.get('/api/positions')
        setPositions(res.data.positions)
        setError(null)
      } catch (err) {
        setError('Failed to load positions')
      } finally {
        setLoading(false)
      }
    }

    fetchPositions()
    const interval = setInterval(fetchPositions, 5000)
    return () => clearInterval(interval)
  }, [])

  // Merge live data
  const displayPositions = liveData.positions.length > 0 ? liveData.positions : positions

  const handleClose = async (ticket) => {
    setClosingTicket(ticket)
    try {
      await api.post(`/api/control/close_position/${ticket}`)
    } catch (err) {
      alert('Failed to close position: ' + err.message)
    } finally {
      setClosingTicket(null)
    }
  }

  const handleCloseAll = async () => {
    if (!window.confirm('Close ALL open positions?')) return
    try {
      await api.post('/api/control/close_all')
    } catch (err) {
      alert('Failed to close all positions: ' + err.message)
    }
  }

  if (loading) return <div className="loading">Loading positions...</div>
  if (error) return <div className="error">{error}</div>

  return (
    <div className="page">
      <div className="page-header">
        <h1>Open Positions</h1>
        {displayPositions.length > 0 && (
          <button className="btn btn-danger" onClick={handleCloseAll} disabled={closingTicket}>
            <X size={16} /> Close All ({displayPositions.length})
          </button>
        )}
      </div>

      {displayPositions.length === 0 ? (
        <div className="card empty-state">
          <AlertTriangle size={48} style={{ color: 'var(--text-muted)', marginBottom: '16px' }} />
          <h3>No Open Positions</h3>
          <p>Positions will appear here when trades are opened.</p>
        </div>
      ) : (
        <div className="card">
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Ticket</th>
                  <th>Direction</th>
                  <th>Volume</th>
                  <th>Entry</th>
                  <th>Current</th>
                  <th>SL</th>
                  <th>TP</th>
                  <th>P/L</th>
                  <th>R-Mult</th>
                  <th>Strategy</th>
                  <th>Duration</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {displayPositions.map(pos => (
                  <tr key={pos.ticket}>
                    <td className="mono">{pos.ticket}</td>
                    <td>
                      <span className={`badge ${pos.direction === 'BUY' ? 'badge-green' : 'badge-red'}`}>
                        {pos.direction}
                      </span>
                    </td>
                    <td className="mono">{pos.volume}</td>
                    <td className="mono">{pos.entry.toFixed(5)}</td>
                    <td className="mono">{pos.current.toFixed(5)}</td>
                    <td className="mono">{pos.sl?.toFixed(5) || 'N/A'}</td>
                    <td className="mono">{pos.tp?.toFixed(5) || 'N/A'}</td>
                    <td className="mono" style={{ color: pos.profit >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                      {pos.profit >= 0 ? '+' : ''}{formatCurrency(pos.profit)}
                    </td>
                    <td className="mono" style={{ color: pos.r_multiple >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                      {pos.r_multiple >= 0 ? '+' : ''}{pos.r_multiple.toFixed(2)}R
                    </td>
                    <td><span className="badge badge-blue">{pos.strategy || 'N/A'}</span></td>
                    <td className="mono">{formatDuration(pos.duration_hours)}</td>
                    <td>
                      <button
                        className="btn btn-danger btn-sm"
                        onClick={() => handleClose(pos.ticket)}
                        disabled={closingTicket === pos.ticket}
                        style={{ padding: '6px 12px', fontSize: '12px' }}
                      >
                        {closingTicket === pos.ticket ? 'Closing...' : 'Close'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}