import React from 'react'
import { Play, Pause, Square, RotateCcw, AlertTriangle, X } from 'lucide-react'
import { formatCurrency, formatNumber, formatPips } from '../utils/format'

export function StatCard({ label, value, icon, trend, trendPositive, positive, warning, children }) {
  return (
    <div className="card">
      <div className="card-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="stat-icon" data-icon={icon} />
          <span className="stat-label">{label}</span>
        </div>
        {warning && <AlertTriangle size={16} className="warning-icon" />}
      </div>
      <div className="stat-value" style={{ color: positive === false ? 'var(--accent-red)' : positive ? 'var(--accent-green)' : 'inherit' }}>
        {value}
      </div>
      {trend && (
        <div className={`stat-trend ${trendPositive ? 'positive' : 'negative'}`}>
          {trendPositive ? '▲' : '▼'} {trend}
        </div>
      )}
      {children}
    </div>
  )
}

export function MarketPanel({ market, tick }) {
  const [price, setPrice] = React.useState(tick?.last || market?.tick?.last || 0)

  React.useEffect(() => {
    if (tick?.last) setPrice(tick.last)
  }, [tick])

  const spread = market?.spread_pips || 0
  const highSpread = market?.high_spread

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Market Monitor</span>
        <span className={`badge ${highSpread ? 'badge-red' : 'badge-green'}`}>
          {highSpread ? 'HIGH SPREAD' : 'NORMAL'}
        </span>
      </div>

      <div style={{ textAlign: 'center', padding: '20px 0' }}>
        <div className="stat-value mono" style={{ fontSize: '36px', marginBottom: '8px' }}>
          {price.toFixed(2)}
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', color: 'var(--text-secondary)', fontSize: '14px' }}>
          <span>Bid: {market?.tick?.bid?.toFixed(2) || 'N/A'}</span>
          <span>Ask: {market?.tick?.ask?.toFixed(2) || 'N/A'}</span>
        </div>
        <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'center', gap: '20px' }}>
          <span className={`badge ${highSpread ? 'badge-red' : 'badge-green'}`}>
            Spread: {formatPips(spread)}
          </span>
          <span className="badge badge-blue">
            Session: {market?.session?.active_session?.toUpperCase() || 'CLOSED'}
          </span>
        </div>
      </div>

      <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '16px', marginTop: '16px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
          <div>
            <div className="stat-label">Floating P/L</div>
            <div className="mono" style={{ color: 'var(--accent-green)' }}>${(market?.account?.floating_pl || 0).toFixed(2)}</div>
          </div>
          <div>
            <div className="stat-label">Free Margin</div>
            <div className="mono">${(market?.account?.free_margin || 0).toFixed(2)}</div>
          </div>
          <div>
            <div className="stat-label">Margin Level</div>
            <div className="mono" style={{ color: (market?.account?.margin_level || 0) < 200 ? 'var(--accent-red)' : 'var(--text-primary)' }}>
              {(market?.account?.margin_level || 0).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export function StructurePanel({ structure }) {
  if (!structure) return <div className="card"><p>No structure data</p></div>

  const tfColors = {
    h4: 'var(--accent-purple)',
    h1: 'var(--accent-blue)',
    m15: 'var(--accent-green)',
    m5: 'var(--accent-yellow)'
  }

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Market Structure</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        {['h4', 'h1', 'm15', 'm5'].map(tf => {
          const s = structure[tf]
          if (!s) return null
          return (
            <div key={tf} style={{
              padding: '16px',
              background: 'var(--bg-tertiary)',
              borderRadius: 'var(--radius-md)',
              borderLeft: `4px solid ${tfColors[tf]}`
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontWeight: 600, color: tfColors[tf], fontSize: '14px' }}>{tf.toUpperCase()}</span>
                <span className={`badge ${s.bias === 'bullish' ? 'badge-green' : s.bias === 'bearish' ? 'badge-red' : 'badge-gray'}`}>
                  {s.bias?.toUpperCase() || 'NEUTRAL'}
                </span>
              </div>
              <div style={{ display: 'flex', gap: '8px', fontSize: '12px' }}>
                <span className={s.bos ? 'badge badge-green' : 'badge badge-gray'}>
                  BOS: {s.bos ? 'YES' : 'NO'}
                </span>
                <span className={s.choch ? 'badge badge-yellow' : 'badge badge-gray'}>
                  CHoCH: {s.choch ? 'YES' : 'NO'}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function LiquidityPanel({ liquidity }) {
  if (!liquidity) return <div className="card"><p>No liquidity data</p></div>

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Liquidity</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ padding: '12px', background: 'rgba(63, 185, 80, 0.1)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(63, 185, 80, 0.2)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>BUY-SIDE</span>
            <span className={liquidity.m15_buy_swept ? 'badge badge-green' : 'badge badge-gray'}>
              {liquidity.m15_buy_swept ? 'SWEPT' : 'ACTIVE'}
            </span>
          </div>
          <div className="mono" style={{ fontSize: '18px' }}>
            {liquidity.h1_buy_levels} Levels (H1)
          </div>
        </div>

        <div style={{ padding: '12px', background: 'rgba(248, 81, 73, 0.1)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(248, 81, 73, 0.2)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ color: 'var(--accent-red)', fontWeight: 600 }}>SELL-SIDE</span>
            <span className={liquidity.m15_sell_swept ? 'badge badge-red' : 'badge badge-gray'}>
              {liquidity.m15_sell_swept ? 'SWEPT' : 'ACTIVE'}
            </span>
          </div>
          <div className="mono" style={{ fontSize: '18px' }}>
            {liquidity.h1_sell_levels} Levels (H1)
          </div>
        </div>
      </div>
    </div>
  )
}

export function OrderBlockPanel({ orderBlocks }) {
  if (!orderBlocks) return <div className="card"><p>No order block data</p></div>

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Order Blocks</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ padding: '12px', background: 'rgba(63, 185, 80, 0.1)', borderRadius: 'var(--radius-md)' }}>
          <div style={{ color: 'var(--accent-green)', fontWeight: 600, marginBottom: '8px' }}>BULLISH OBs</div>
          <div className="mono" style={{ fontSize: '24px' }}>{orderBlocks.m15_bullish || 0} M15</div>
          <div className="mono" style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>{orderBlocks.m5_bullish || 0} M5</div>
        </div>

        <div style={{ padding: '12px', background: 'rgba(248, 81, 73, 0.1)', borderRadius: 'var(--radius-md)' }}>
          <div style={{ color: 'var(--accent-red)', fontWeight: 600, marginBottom: '8px' }}>BEARISH OBs</div>
          <div className="mono" style={{ fontSize: '24px' }}>{orderBlocks.m15_bearish || 0} M15</div>
          <div className="mono" style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>{orderBlocks.m5_bearish || 0} M5</div>
        </div>
      </div>
    </div>
  )
}

export function FVGPanel({ fvgs }) {
  if (!fvgs) return <div className="card"><p>No FVG data</p></div>

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Fair Value Gaps</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ padding: '12px', background: 'rgba(63, 185, 80, 0.1)', borderRadius: 'var(--radius-md)' }}>
          <div style={{ color: 'var(--accent-green)', fontWeight: 600, marginBottom: '8px' }}>BULLISH FVGs</div>
          <div className="mono" style={{ fontSize: '24px' }}>{fvgs.m5_bullish || 0} M5</div>
          <div className="mono" style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>{fvgs.m15_bullish || 0} M15</div>
        </div>

        <div style={{ padding: '12px', background: 'rgba(248, 81, 73, 0.1)', borderRadius: 'var(--radius-md)' }}>
          <div style={{ color: 'var(--accent-red)', fontWeight: 600, marginBottom: '8px' }}>BEARISH FVGs</div>
          <div className="mono" style={{ fontSize: '24px' }}>{fvgs.m5_bearish || 0} M5</div>
          <div className="mono" style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>{fvgs.m15_bearish || 0} M15</div>
        </div>
      </div>
    </div>
  )
}

export function PremiumDiscountPanel({ premiumDiscount }) {
  if (!premiumDiscount) return <div className="card"><p>No premium/discount data</p></div>

  const zoneColors = {
    premium: 'var(--accent-red)',
    discount: 'var(--accent-green)',
    equilibrium: 'var(--accent-yellow)',
    unknown: 'var(--text-secondary)'
  }

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Premium / Discount</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        {['h4', 'h1', 'm15', 'm5'].map(tf => {
          const zone = premiumDiscount[tf] || 'unknown'
          return (
            <div key={tf} style={{
              padding: '16px',
              background: 'var(--bg-tertiary)',
              borderRadius: 'var(--radius-md)',
              textAlign: 'center',
              borderTop: `3px solid ${zoneColors[zone]}`
            }}>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px', textTransform: 'uppercase' }}>{tf}</div>
              <span className={`badge`} style={{
                background: `${zoneColors[zone]}20`,
                color: zoneColors[zone],
                fontSize: '14px',
                padding: '8px 16px'
              }}>
                {zone.toUpperCase()}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function ControlPanel({ positions, account, botState }) {
  const [actionLoading, setActionLoading] = React.useState(null)

  const handleAction = async (action) => {
    setActionLoading(action)
    try {
      const api = (await import('../services/api')).api
      if (action === 'pause') {
        await api.post('/api/control/pause')
      } else if (action === 'resume') {
        await api.post('/api/control/resume')
      } else if (action === 'close_all') {
        await api.post('/api/control/close_all')
      } else if (action === 'emergency') {
        await api.post('/api/control/emergency_stop')
      } else if (action === 'shutdown') {
        await api.post('/api/control/shutdown')
      }
      // Refresh will happen via WebSocket
    } catch (err) {
      console.error('Action failed:', err)
      alert('Action failed: ' + err.message)
    } finally {
      setActionLoading(null)
    }
  }

  const isRunning = botState === 'running'
  const isPaused = botState === 'paused'

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Trading Controls</span>
        <span className={`badge ${isRunning ? 'badge-green' : isPaused ? 'badge-yellow' : 'badge-red'}`}>
          {isRunning ? 'RUNNING' : isPaused ? 'PAUSED' : 'STOPPED'}
        </span>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', marginBottom: '20px' }}>
        {!isRunning && !isPaused && (
          <>
            <button
              className="btn btn-primary"
              onClick={() => handleAction('resume')}
              disabled={actionLoading === 'resume'}
            >
              <Play size={16} /> Start Trading
            </button>
          </>
        )}

        {isRunning && (
          <>
            <button
              className="btn btn-secondary"
              onClick={() => handleAction('pause')}
              disabled={actionLoading === 'pause'}
            >
              <Pause size={16} /> Pause
            </button>
            <button
              className="btn btn-danger"
              onClick={() => handleAction('close_all')}
              disabled={actionLoading === 'close_all' || positions.length === 0}
            >
              <X size={16} /> Close All ({positions.length})
            </button>
          </>
        )}

        {isPaused && (
          <>
            <button
              className="btn btn-primary"
              onClick={() => handleAction('resume')}
              disabled={actionLoading === 'resume'}
            >
              <Play size={16} /> Resume
            </button>
          </>
        )}

        <button
          className="btn btn-danger"
          onClick={() => handleAction('emergency')}
          disabled={actionLoading === 'emergency'}
          style={{ marginLeft: 'auto', background: 'var(--accent-red)', border: '2px solid var(--accent-red)' }}
        >
          <AlertTriangle size={16} /> EMERGENCY STOP
        </button>

        <button
          className="btn btn-secondary"
          onClick={() => handleAction('shutdown')}
          disabled={actionLoading === 'shutdown'}
        >
          <RotateCcw size={16} /> Shutdown
        </button>
      </div>

      {/* Positions Summary */}
      {positions.length > 0 && (
        <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '16px' }}>
          <div style={{ fontWeight: 600, marginBottom: '12px' }}>Open Positions ({positions.length})</div>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Dir</th>
                  <th>Volume</th>
                  <th>Entry</th>
                  <th>Current</th>
                  <th>P/L</th>
                  <th>R</th>
                </tr>
              </thead>
              <tbody>
                {positions.slice(0, 5).map((pos, i) => (
                  <tr key={pos.ticket}>
                    <td className="mono">{pos.ticket}</td>
                    <td><span className={`badge ${pos.direction === 'BUY' ? 'badge-green' : 'badge-red'}`}>{pos.direction}</span></td>
                    <td className="mono">{pos.volume}</td>
                    <td className="mono">{pos.entry.toFixed(2)}</td>
                    <td className="mono">{pos.current.toFixed(2)}</td>
                    <td className="mono" style={{ color: pos.profit >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                      {pos.profit >= 0 ? '+' : ''}{pos.profit.toFixed(2)}
                    </td>
                    <td className="mono">{pos.r_multiple >= 0 ? '+' : ''}{pos.r_multiple.toFixed(2)}R</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {positions.length > 5 && (
            <p style={{ marginTop: '8px', color: 'var(--text-secondary)', fontSize: '13px' }}>
              + {positions.length - 5} more positions
            </p>
          )}
        </div>
      )}
    </div>
  )
}