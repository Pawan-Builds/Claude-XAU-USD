import React from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { Menu, X, ChevronLeft, ChevronRight, Circle, Wifi, WifiOff, AlertCircle, CheckCircle } from 'lucide-react'
import './Layout.css'

export function Layout({ children, sidebarOpen, setSidebarOpen }) {
  return (
    <div className={`app-layout ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
      <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <h1>XAU/USD SMC</h1>
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {sidebarOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
          </button>
        </div>
        <nav className="sidebar-nav">
          <Outlet />
        </nav>
        <div className="sidebar-footer">
          <div className="version">v1.0.0</div>
        </div>
      </aside>
      <div className="main-content">
        {children}
      </div>
    </div>
  )
}

export function Header({ systemStatus, onMenuClick }) {
  const getStatusColor = (state) => {
    switch (state) {
      case 'running': return 'status-running'
      case 'paused': return 'status-paused'
      case 'stopped': return 'status-stopped'
      case 'emergency': return 'status-emergency'
      default: return 'status-unknown'
    }
  }

  const getStatusLabel = (state) => {
    switch (state) {
      case 'running': return 'RUNNING'
      case 'paused': return 'PAUSED'
      case 'stopped': return 'STOPPED'
      case 'emergency': return 'EMERGENCY'
      default: return 'UNKNOWN'
    }
  }

  return (
    <header className="header">
      <button className="mobile-menu-btn" onClick={onMenuClick}>
        <Menu size={24} />
      </button>
      <div className="header-title">
        <h2>XAU/USD SMC Trading Dashboard</h2>
      </div>
      <div className="header-status">
        {systemStatus && (
          <div className={`status-badge ${getStatusColor(systemStatus.bot_state)}`}>
            <Circle className="status-dot" size={8} />
            <span>{getStatusLabel(systemStatus.bot_state)}</span>
          </div>
        )}
        {systemStatus && (
          <div className={`connection-status ${systemStatus.mt5_connected ? 'connected' : 'disconnected'}`}>
            {systemStatus.mt5_connected ? <Wifi size={16} /> : <WifiOff size={16} />}
            <span>{systemStatus.mt5_connected ? 'MT5 Connected' : 'MT5 Disconnected'}</span>
          </div>
        )}
        {systemStatus && systemStatus.account && (
          <div className="account-info">
            <span>${systemStatus.account.equity?.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>
            <span className="account-label">Equity</span>
          </div>
        )}
      </div>
    </header>
  )
}

export function Sidebar({ open, items }) {
  return (
    <ul className="sidebar-menu">
      {items.map((item) => (
        <li key={item.path}>
          <NavLink
            to={item.path}
            className={({ isActive }) => `menu-item ${isActive ? 'active' : ''}`}
            onClick={() => {}}
          >
            <span className="menu-icon" data-icon={item.icon} />
            <span className="menu-label">{item.label}</span>
          </NavLink>
        </li>
      ))}
    </ul>
  )
}

export function Content({ children }) {
  return (
    <main className="content">
      {children}
    </main>
  )
}

export function Footer() {
  return (
    <footer className="footer">
      <span>XAU/USD SMC Trading System</span>
      <span>Demo Mode Only</span>
    </footer>
  )
}