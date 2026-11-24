import { useEffect, useRef, useState } from 'react'

const TOKEN_KEY = 'dashboard_token'

function withToken(url) {
  const token = localStorage.getItem(TOKEN_KEY)
  if (!token) return null
  const sep = url.includes('?') ? '&' : '?'
  return `${url}${sep}token=${encodeURIComponent(token)}`
}

export function useWebSocket(url, options = {}) {
  const [lastMessage, setLastMessage] = useState(null)
  const [connectionStatus, setConnectionStatus] = useState('connecting')
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const reconnectAttempts = useRef(0)
  const maxReconnectAttempts = 10

  const { onMessage, onOpen, onClose, onError } = options

  useEffect(() => {
    let mounted = true

    const connect = () => {
      if (!mounted) return

      // Wait until the HTTP layer has prompted the user for a token.
      const authedUrl = withToken(url)
      if (!authedUrl) {
        setConnectionStatus('waiting_for_token')
        reconnectTimeoutRef.current = setTimeout(connect, 2000)
        return
      }

      try {
        const ws = new WebSocket(authedUrl)
        wsRef.current = ws

        ws.onopen = () => {
          if (!mounted) return
          setConnectionStatus('connected')
          reconnectAttempts.current = 0
          onOpen?.()
        }

        ws.onmessage = (event) => {
          if (!mounted) return
          try {
            const data = JSON.parse(event.data)
            setLastMessage(data)
            onMessage?.(data)
          } catch (err) {
            console.error('WS parse error:', err)
          }
        }

        ws.onclose = () => {
          if (!mounted) return
          setConnectionStatus('disconnected')
          onClose?.()

          if (reconnectAttempts.current < maxReconnectAttempts) {
            const delay = Math.min(
              1000 * Math.pow(2, reconnectAttempts.current),
              30000
            )
            reconnectTimeoutRef.current = setTimeout(() => {
              reconnectAttempts.current++
              connect()
            }, delay)
          }
        }

        ws.onerror = (error) => {
          if (!mounted) return
          setConnectionStatus('error')
          onError?.(error)
        }
      } catch (err) {
        console.error('WS connection error:', err)
        setConnectionStatus('error')
      }
    }

    connect()

    return () => {
      mounted = false
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [url])

  const send = (data) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }

  return { lastMessage, connectionStatus, send }
}