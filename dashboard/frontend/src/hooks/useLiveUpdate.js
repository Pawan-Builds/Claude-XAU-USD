import { useState, useEffect } from 'react'
import { useWebSocket } from '../services/websocket'

export function useLiveUpdate() {
  const [liveData, setLiveData] = useState({
    tick: null,
    account: null,
    positions: [],
    botState: 'stopped'
  })

  const { lastMessage } = useWebSocket('ws://127.0.0.1:8000/ws', {
    onMessage: (data) => {
      if (data.type === 'live_update') {
        setLiveData(prev => ({
          ...prev,
          tick: data.tick,
          account: data.account,
          positions: data.positions,
          botState: data.bot_state
        }))
      }
    }
  })

  return liveData
}