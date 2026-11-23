import React, { useState, useEffect } from 'react'
import { useLiveUpdate } from '../hooks/useLiveUpdate'
import { api } from '../services/api'
import { StatCard, MarketPanel, StructurePanel, LiquidityPanel, OrderBlockPanel, FVGPanel, PremiumDiscountPanel, ControlPanel } from '../components/DashboardComponents'
import { formatCurrency, formatNumber } from '../utils/format'

export default function Dashboard() {
  const [account, setAccount] = useState(null)
  const [market, setMarket] = useState(null)
  const [performance, setPerformance] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Live updates from WebSocket
  const liveData = useLiveUpdate()

  // Fetch initial data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const [accRes, mktRes, perfRes] = await Promise.all([
          api.get('/api/account'),
          api.get('/api/market'),
          api.get('/api/performance')
        ])
        setAccount(accRes.data)
        setMarket(mktRes.data)
        setPerformance(perfRes.data)
        setError(null)
      } catch (err) {
        setError('Failed to load dashboard data')
        console.error(err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 30000)
    return () => clearInterval(interval)
  }, [])

  // Merge live data with fetched data
  const displayAccount = liveData.account || account
  const displayMarket = liveData.market || market
  const displayPositions = liveData.positions || []

  if (loading && !account) {
    return <div className="loading">Loading dashboard...</div>
  }

  if (error) {
    return <div className="error">{error}</div>
  }

  return (
    <div className="dashboard">
      {/* Top Stats Row */}
      <div className="grid grid-5">
        <StatCard
          label="Balance"
          value={displayAccount ? formatCurrency(displayAccount.balance) : '$0.00'}
          icon="wallet"
        />
        <StatCard
          label="Equity"
          value={displayAccount ? formatCurrency(displayAccount.equity) : '$0.00'}
          icon="trending-up"
          trend={displayAccount && displayAccount.floating_pl ? formatCurrency(displayAccount.floating_pl) : null}
          trendPositive={displayAccount && displayAccount.floating_pl >= 0}
        />
        <StatCard
          label="Free Margin"
          value={displayAccount ? formatCurrency(displayAccount.free_margin) : '$0.00'}
          icon="credit-card"
        />
        <StatCard
          label="Margin Level"
          value={displayAccount ? `${displayAccount.margin_level?.toFixed(1) || 0}%` : '0%'}
          icon="gauge"
          warning={displayAccount && displayAccount.margin_level < 200}
        />
        <StatCard
          label="Daily P/L"
          value={displayAccount ? formatCurrency(displayAccount.daily_pl || 0) : '$0.00'}
          icon="calendar"
          trendPositive={displayAccount && (displayAccount.daily_pl || 0) >= 0}
        />
      </div>

      {/* Main Grid */}
      <div className="grid grid-2" style={{ marginTop: '20px' }}>
        {/* Market Panel */}
        <MarketPanel market={displayMarket} tick={liveData.tick} />

        {/* Structure Panel */}
        <StructurePanel structure={displayMarket?.structure} />
      </div>

      <div className="grid grid-3" style={{ marginTop: '20px' }}>
        <LiquidityPanel liquidity={displayMarket?.liquidity} />
        <OrderBlockPanel orderBlocks={displayMarket?.order_blocks} />
        <FVGPanel fvgs={displayMarket?.fvgs} />
      </div>

      <div className="grid grid-2" style={{ marginTop: '20px' }}>
        <PremiumDiscountPanel premiumDiscount={displayMarket?.premium_discount} />
        <ControlPanel
          positions={displayPositions}
          account={displayAccount}
          botState={liveData.botState}
        />
      </div>

      {/* Performance Summary */}
      {performance && performance.total_trades > 0 && (
        <div className="grid grid-4" style={{ marginTop: '20px' }}>
          <StatCard
            label="Total Trades"
            value={performance.total_trades}
            icon="list"
          />
          <StatCard
            label="Win Rate"
            value={`${performance.win_rate}%`}
            icon="target"
            positive={performance.win_rate >= 50}
          />
          <StatCard
            label="Profit Factor"
            value={typeof performance.profit_factor === 'number' ? performance.profit_factor.toFixed(2) : performance.profit_factor}
            icon="ratio"
            positive={typeof performance.profit_factor === 'number' && performance.profit_factor > 1}
          />
          <StatCard
            label="Net Profit"
            value={formatCurrency(performance.net_profit)}
            icon="dollar-sign"
            trendPositive={performance.net_profit >= 0}
          />
        </div>
      )}
    </div>
  )
}