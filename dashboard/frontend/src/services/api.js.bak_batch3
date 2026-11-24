import axios from 'axios'

const TOKEN_KEY = 'dashboard_token'

function getToken() {
  let token = localStorage.getItem(TOKEN_KEY)
  if (!token) {
    token = window.prompt('Enter dashboard token (DASHBOARD_SECRET_KEY):')
    if (token) {
      token = token.trim()
      localStorage.setItem(TOKEN_KEY, token)
    }
  }
  return token || null
}

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
})

api.interceptors.request.use(config => {
  const token = getToken()
  if (!token) {
    return Promise.reject(new Error('Dashboard token required'))
  }
  config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      localStorage.removeItem(TOKEN_KEY)
    }
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

export { api }