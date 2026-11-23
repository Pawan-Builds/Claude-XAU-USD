import React, { useState, useEffect } from 'react'
import { api } from '../services/api'
import { Save, RotateCcw, AlertTriangle, CheckCircle } from 'lucide-react'

export default function Settings() {
  const [riskConfig, setRiskConfig] = useState({
    risk_percent: 2.0,
    max_positions: 3,
    max_daily_loss: 5.0,
    max_drawdown: 10.0,
    compound_enabled: true
  })
  const [sessionConfig, setSessionConfig] = useState({
    london_enabled: true,
    london_open: '08:00',
    london_close: '17:00',
    ny_enabled: true,
    ny_open: '13:00',
    ny_close: '22:00',
    asian_enabled: false,
    asian_open: '00:00',
    asian_close: '09:00',
    weekend_protection: true
  })
  const [tradeConfig, setTradeConfig] = useState({
    break_even_enabled: true,
    break_even_trigger_r: 1.0,
    trailing_enabled: true,
    trailing_start_r: 1.5,
    partial_tp_enabled: true
  })
  const [strategyConfig, setStrategyConfig] = useState({
    strategies: ['liquidity_sweep_fvg'],
    min_risk_reward: 2.0
  })
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState(null) // 'success', 'error', null

  const strategyOptions = [
    { value: 'liquidity_sweep_fvg', label: 'Liquidity Sweep + FVG' },
    { value: 'liquidity_sweep_choch', label: 'Liquidity Sweep + CHoCH' },
    { value: 'bos_order_block', label: 'BOS + Order Block' },
    { value: 'choch_order_block', label: 'CHoCH + Order Block' },
    { value: 'displacement_fvg', label: 'Displacement + FVG' },
    { value: 'liquidity_sweep_ob_fvg', label: 'Liquidity Sweep + OB + FVG' }
  ]

  useEffect(() => {
    // In a real implementation, fetch current config from backend
    setLoading(false)
  }, [])

  const handleSave = async (section) => {
    setSaving(true)
    setSaveStatus(null)
    try {
      // In a real implementation, POST to /api/settings
      await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate API call
      setSaveStatus('success')
    } catch (err) {
      setSaveStatus('error')
    } finally {
      setSaving(false)
      setTimeout(() => setSaveStatus(null), 3000)
    }
  }

  const handleRiskChange = (field, value) => {
    setRiskConfig(prev => ({ ...prev, [field]: value }))
  }

  const handleSessionChange = (field, value) => {
    setSessionConfig(prev => ({ ...prev, [field]: value }))
  }

  const handleTradeChange = (field, value) => {
    setTradeConfig(prev => ({ ...prev, [field]: value }))
  }

  const handleStrategyChange = (field, value) => {
    setStrategyConfig(prev => ({ ...prev, [field]: value }))
  }

  const handleMultiStrategy = (value) => {
    setStrategyConfig(prev => {
      const current = prev.strategies || []
      const updated = current.includes(value)
        ? current.filter(v => v !== value)
        : [...current, value]
      return { ...prev, strategies: updated }
    })
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1>Settings</h1>
      </div>

      {saveStatus && (
        <div className={`alert ${saveStatus}`}>
          {saveStatus === 'success' ? (
            <>
              <CheckCircle size={20} /> Settings saved successfully
            </>
          ) : (
            <>
              <AlertTriangle size={20} /> Failed to save settings
            </>
          )}
        </div>
      )}

      {/* Risk Management */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="card-header">
          <span className="card-title">Risk Management</span>
          <button className="btn btn-primary btn-sm" onClick={() => handleSave('risk')} disabled={saving}>
            <Save size={16} /> {saving ? 'Saving...' : 'Save Risk Config'}
          </button>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px' }}>
          <div>
            <label className="stat-label">Risk per Trade (%)</label>
            <input className="input" type="number" step="0.1" min="0.1" max="5"
              value={riskConfig.risk_percent} onChange={e => handleRiskChange('risk_percent', parseFloat(e.target.value))} />
          </div>
          <div>
            <label className="stat-label">Max Open Positions</label>
            <input className="input" type="number" min="1" max="10"
              value={riskConfig.max_positions} onChange={e => handleRiskChange('max_positions', parseInt(e.target.value))} />
          </div>
          <div>
            <label className="stat-label">Max Daily Loss (%)</label>
            <input className="input" type="number" step="0.1" min="1" max="20"
              value={riskConfig.max_daily_loss} onChange={e => handleRiskChange('max_daily_loss', parseFloat(e.target.value))} />
          </div>
          <div>
            <label className="stat-label">Max Account Drawdown (%)</label>
            <input className="input" type="number" step="0.1" min="5" max="50"
              value={riskConfig.max_drawdown} onChange={e => handleRiskChange('max_drawdown', parseFloat(e.target.value))} />
          </div>
          <div>
            <label className="stat-label" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input type="checkbox" checked={riskConfig.compound_enabled} onChange={e => handleRiskChange('compound_enabled', e.target.checked)} />
              Enable Equity Compounding
            </label>
          </div>
        </div>
      </div>

      {/* Trading Sessions */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="card-header">
          <span className="card-title">Trading Sessions (UTC)</span>
          <button className="btn btn-primary btn-sm" onClick={() => handleSave('sessions')} disabled={saving}>
            <Save size={16} /> {saving ? 'Saving...' : 'Save Session Config'}
          </button>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
          {[
            { key: 'london', label: 'London', enabled: sessionConfig.london_enabled, open: sessionConfig.london_open, close: sessionConfig.london_close },
            { key: 'new_york', label: 'New York', enabled: sessionConfig.ny_enabled, open: sessionConfig.ny_open, close: sessionConfig.ny_close },
            { key: 'asian', label: 'Asian', enabled: sessionConfig.asian_enabled, open: sessionConfig.asian_open, close: sessionConfig.asian_close }
          ].map(s => (
            <div key={s.key} style={{ padding: '16px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', border: `1px solid ${sessionConfig[`${s.key}_enabled`] ? 'var(--accent-blue)' : 'var(--border-color)'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                  <input type="checkbox"
                    checked={sessionConfig[`${s.key}_enabled`]}
                    onChange={e => handleSessionChange(`${s.key}_enabled`, e.target.checked)} />
                  <strong>{s.label} Session</strong>
                </label>
              </div>
              {sessionConfig[`${s.key}_enabled`] && (
                <div style={{ display: 'flex', gap: '16px' }}>
                  <div>
                    <label className="stat-label">Open</label>
                    <input className="input" type="time"
                      value={sessionConfig[`${s.key}_open`]}
                      onChange={e => handleSessionChange(`${s.key}_open`, e.target.value)} />
                  </div>
                  <div>
                    <label className="stat-label">Close</label>
                    <input className="input" type="time"
                      value={sessionConfig[`${s.key}_close`]}
                      onChange={e => handleSessionChange(`${s.key}_close`, e.target.value)} />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
        <div style={{ marginTop: '16px' }}>
          <label className="stat-label" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input type="checkbox" checked={sessionConfig.weekend_protection} onChange={e => handleSessionChange('weekend_protection', e.target.checked)} />
            Weekend Protection (close positions before weekend)
          </label>
        </div>
      </div>

      {/* Trade Management */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="card-header">
          <span className="card-title">Trade Management</span>
          <button className="btn btn-primary btn-sm" onClick={() => handleSave('trade')} disabled={saving}>
            <Save size={16} /> {saving ? 'Saving...' : 'Save Trade Config'}
          </button>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px' }}>
          <div>
            <label className="stat-label" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input type="checkbox" checked={tradeConfig.break_even_enabled} onChange={e => handleTradeChange('break_even_enabled', e.target.checked)} />
              Enable Break-Even
            </label>
          </div>
          {tradeConfig.break_even_enabled && (
            <div>
              <label className="stat-label">Break-Even Trigger (R)</label>
              <input className="input" type="number" step="0.1" min="0.5" max="3"
                value={tradeConfig.break_even_trigger_r} onChange={e => handleTradeChange('break_even_trigger_r', parseFloat(e.target.value))} />
            </div>
          )}
          <div>
            <label className="stat-label" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input type="checkbox" checked={tradeConfig.trailing_enabled} onChange={e => handleTradeChange('trailing_enabled', e.target.checked)} />
              Enable Trailing Stop
            </label>
          </div>
          {tradeConfig.trailing_enabled && (
            <div>
              <label className="stat-label">Trailing Start (R)</label>
              <input className="input" type="number" step="0.1" min="0.5" max="5"
                value={tradeConfig.trailing_start_r} onChange={e => handleTradeChange('trailing_start_r', parseFloat(e.target.value))} />
            </div>
          )}
          <div>
            <label className="stat-label" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input type="checkbox" checked={tradeConfig.partial_tp_enabled} onChange={e => handleTradeChange('partial_tp_enabled', e.target.checked)} />
              Enable Partial TP
            </label>
          </div>
        </div>
      </div>

      {/* Strategies */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="card-header">
          <span className="card-title">SMC Strategies</span>
          <button className="btn btn-primary btn-sm" onClick={() => handleSave('strategies')} disabled={saving}>
            <Save size={16} /> {saving ? 'Saving...' : 'Save Strategy Config'}
          </button>
        </div>
        <div style={{ marginBottom: '16px' }}>
          <label className="stat-label">Minimum Risk/Reward</label>
          <input className="input" type="number" step="0.1" min="1" max="5" style={{ maxWidth: '120px' }}
            value={strategyConfig.min_risk_reward} onChange={e => handleStrategyChange('min_risk_reward', parseFloat(e.target.value))} />
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
          {strategyOptions.map(s => (
            <label key={s.value} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '12px 16px', background: strategyConfig.strategies.includes(s.value) ? 'rgba(88, 166, 255, 0.15)' : 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', border: `1px solid ${strategyConfig.strategies.includes(s.value) ? 'var(--accent-blue)' : 'var(--border-color)'}`, cursor: 'pointer' }}>
              <input type="checkbox"
                checked={strategyConfig.strategies.includes(s.value)}
                onChange={e => handleMultiStrategy(s.value)} />
              <span>{s.label}</span>
            </label>
          ))}
        </div>
      </div>

      {/* System Info */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">System Information</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px' }}>
          <div style={{ padding: '16px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
            <div className="stat-label">Version</div>
            <div className="stat-value mono" style={{ fontSize: '18px' }}>1.0.0</div>
          </div>
          <div style={{ padding: '16px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
            <div className="stat-label">Mode</div>
            <div className="stat-value" style={{ color: 'var(--accent-yellow)' }}>DEMO ONLY</div>
          </div>
          <div style={{ padding: '16px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
            <div className="stat-label">MT5 Path</div>
            <div className="stat-value mono" style={{ fontSize: '12px' }}>C:\\Program Files\\XAUUSD\\terminal64.exe</div>
          </div>
          <div style={{ padding: '16px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
            <div className="stat-label">Database</div>
            <div className="stat-value mono" style={{ fontSize: '12px' }}>SQLite (database/trading.db)</div>
          </div>
        </div>
      </div>
    </div>
  )
}