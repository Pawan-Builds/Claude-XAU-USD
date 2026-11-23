import React, { useState, useEffect } from 'react'
import { Routes, Route, Link, useNavigate } from 'react-router-dom'
import { Layout, Header, Sidebar, Content, Footer } from './components/Layout'
import Dashboard from './pages/Dashboard'
import Positions from './pages/Positions'
import History from './pages/History'
import Performance from './pages/Performance'
import Backtesting from './pages/Backtesting'
import Settings from './pages/Settings'
import { useWebSocket } from './services/websocket'
import { api } from './services/api'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [systemStatus, setSystemStatus] = useState(null)
  const navigate = useNavigate()

  // WebSocket for live updates
  const { lastMessage } = useWebSocket('ws://127.0.0.1:8000/ws', {
    onMessage: (data) => {
      if (data.type === 'live_update') {
        // Update global state with live data
        window.dispatchEvent(new CustomEvent('live-update', { detail: data }))
      }
    }
  })

  // Fetch initial status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await api.get('/api/status')
        setSystemStatus(res.data)
      } catch (err) {
        console.error('Failed to fetch status:', err)
      }
    }
    fetchStatus()
    const interval = setInterval(fetchStatus, 10000)
    return () => clearInterval(interval)
  }, [])

  const menuItems = [
    { path: '/', label: 'Dashboard', icon: 'layout-dashboard' },
    { path: '/positions', label: 'Positions', icon: 'chart-candlestick' },
    { path: '/history', label: 'History', icon: 'history' },
    { path: '/performance', label: 'Performance', icon: 'bar-chart-2' },
    { path: '/backtesting', label: 'Backtesting', icon: 'flask-conical' },
    { path: '/settings', label: 'Settings', icon: 'settings' }
  ]

  return (
    <Layout sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen}>
      <Header systemStatus={systemStatus} onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
      <Sidebar open={sidebarOpen} items={menuItems} />
      <Content>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/positions" element={<Positions />} />
          <Route path="/history" element={<History />} />
          <Route path="/performance" element={<Performance />} />
          <Route path="/backtesting" element={<Backtesting />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Content>
      <Footer />
    </Layout>
  )
}

export default App