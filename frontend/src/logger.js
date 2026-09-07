// logger.js
import { BACKEND_URL } from './config'

// Optional: global logging context (task_name, chart_mode, chart_task, etc.)
let __logCtx = {};
export function setLogContext(ctx = {}) {
  __logCtx = { ...__logCtx, ...ctx };
}

// Generate or fetch a persistent participant/user id for this browser
export function getUserId() {
  let id = localStorage.getItem('participant_id')
  if (!id) {
    id = `u-${Math.random().toString(36).slice(2, 6)}${Date.now().toString(36).slice(-4)}`
    localStorage.setItem('participant_id', id)
  }
  return id
}

// One-per-tab session id (resets on page refresh)
export function getSessionId() {
  let sid = sessionStorage.getItem('session_id')
  if (!sid) {
    sid = `s-${Math.random().toString(36).slice(2, 8)}${Date.now().toString(36).slice(-4)}`
    sessionStorage.setItem('session_id', sid)
  }
  return sid
}

// Keep your existing signature. Optional third arg for extra fields if needed.
export async function logEvent(eventType, detail, extra = {}) {
  const user_id = getUserId()
  const session_id = getSessionId()
  const ts = new Date().toISOString()
  const path = typeof window !== 'undefined'
    ? `${window.location.pathname}${window.location.search}`
    : ''

  if (!eventType) return // small guard

  const payload = {
    eventType,
    detail,
    path,                 // <-- use the full path you computed
    client_ts: ts,        // optional but handy
    ...__logCtx,
    ...extra,
  }

  try {
    await fetch(`${BACKEND_URL}/log`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-User-Id': user_id,
        'X-Session-Id': session_id,   // <-- include session id
      },
      body: JSON.stringify(payload),
      keepalive: true,
    })
  } catch (error) {
    console.error('Error logging event:', error)
  }
}