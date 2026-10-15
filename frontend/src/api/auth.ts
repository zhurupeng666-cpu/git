// 账户角色和用户信息类型
export type UserRole = 'employee' | 'engineer'
export interface UserProfile {
  user_id: string
  username: string
  display_name: string
  role: UserRole
  csrf_token: string
  session_id: string
  idle_timeout_seconds: number
}
export interface LoginResponse extends UserProfile {
  tab_token: string
}

// CSRF Token只保存在当前页面内存中，JWT由浏览器通过HttpOnly Cookie管理
let csrfToken = ''
// sid是公开会话标识，不是JWT；sessionStorage按标签页隔离并支持刷新恢复。
const sessionKey = 'semiconductor_tab_credentials'
let sessionId = ''
let userId = ''
let tabToken = ''
let idleSeconds = 1800
let lastActivity = Date.now()
let lastActivityRequest = 0
try {
  const saved = JSON.parse(sessionStorage.getItem(sessionKey) || 'null')
  if (
    typeof saved?.session_id === 'string' &&
    typeof saved?.tab_token === 'string' &&
    saved.tab_token
  ) {
    sessionId = saved.session_id
    tabToken = saved.tab_token
  }
  // 旧版只有公开sid，不能当作标签页凭证使用。
  sessionStorage.removeItem('semiconductor_page_session')
} catch {
  /* 无存储权限时仍在页面内存中隔离登录 */
}
function bindAuthState(user: UserProfile, token = tabToken): void {
  csrfToken = user.csrf_token
  sessionId = user.session_id
  userId = user.user_id
  tabToken = token
  idleSeconds = user.idle_timeout_seconds || 1800
  try {
    if (sessionId && tabToken)
      sessionStorage.setItem(
        sessionKey,
        JSON.stringify({ session_id: sessionId, tab_token: tabToken }),
      )
  } catch {
    /* 不影响HttpOnly Cookie认证 */
  }
}
export function getAuthScope(): { user_id: string; session_id: string } {
  return { user_id: userId, session_id: sessionId }
}
interface AuthNotice {
  type: 'login-changed' | 'logout'
  user_id: string
  session_id: string
}
function receiveAuthNotice(data: unknown): void {
  if (!data || typeof data !== 'object') return
  const notice = data as Partial<AuthNotice>
  if (!userId || notice.user_id !== userId || !notice.session_id || !sessionId) return
  // 只退出同账号被替换的会话，或已明确注销的同一个会话。
  if (
    (notice.type === 'login-changed' && notice.session_id !== sessionId) ||
    (notice.type === 'logout' && notice.session_id === sessionId)
  )
    notifyAuthExpired()
}

// 同源标签页同步退出，不在浏览器存储JWT或CSRF Token
const logoutKey = 'semiconductor_logout_event'
// BroadcastChannel即时同步退出；保留storage作为旧浏览器兼容。
const authChannel =
  typeof window.BroadcastChannel === 'function'
    ? new window.BroadcastChannel('semiconductor-auth')
    : null
if (authChannel) authChannel.onmessage = (event) => receiveAuthNotice(event.data)
export const AUTH_EXPIRED_EVENT = 'semiconductor-auth-expired'
export function notifyAuthExpired(): void {
  clearAuthState()
  window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT))
}
window.addEventListener('storage', (event) => {
  if (event.key !== logoutKey || !event.newValue) return
  try {
    receiveAuthNotice(JSON.parse(event.newValue))
  } catch {
    /* 忽略旧版本和无账号范围的通知，不能强制退出所有页面 */
  }
})

// 用户登录：后端通过Set-Cookie写入JWT，前端只接收用户信息与CSRF Token
export async function login(username: string, password: string): Promise<UserProfile> {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!response.ok)
    throw new Error(response.status === 401 ? '用户名或密码错误' : '登录失败，请稍后重试')
  const user: LoginResponse = await response.json()
  if (!user.tab_token) throw new Error('登录接口未返回标签页凭证，请检查后端版本')
  const { tab_token, ...profile } = user
  bindAuthState(profile, tab_token)
  lastActivity = Date.now()
  lastActivityRequest = 0
  // 通知带账号和会话范围，不同账号的页面不受影响。
  authChannel?.postMessage({ type: 'login-changed', ...getAuthScope() })
  return profile
}

// 获取当前用户：由浏览器自动携带HttpOnly Cookie
export async function getCurrentUser(signal?: AbortSignal): Promise<UserProfile> {
  // 新标签页没有独立凭证时直接显示登录页，不能尝试共享Cookie恢复。
  if (!sessionId || !tabToken) throw new Error('登录已失效，请重新登录')
  const requestedSession = sessionId
  const headers = new Headers()
  if (requestedSession) headers.set('X-Session-ID', requestedSession)
  headers.set('X-Tab-Token', tabToken)
  const response = await fetch('/api/auth/me', { credentials: 'same-origin', headers, signal })
  // 旧请求返回时页面可能已经登录另一个账号，不能覆盖或清理新身份。
  if (requestedSession !== sessionId) throw new Error('请求期间登录已变更，请重试')
  if (!response.ok) {
    if (response.status === 401) notifyAuthExpired()
    throw new Error(response.status === 401 ? '登录已失效，请重新登录' : '获取用户信息失败')
  }
  const user: UserProfile = await response.json()
  if (sessionId && sessionId !== user.session_id) {
    notifyAuthExpired()
    throw new Error('登录已失效，请重新登录')
  }
  bindAuthState(user)
  return user
}

// 获取CSRF Token，页面刷新后自动通过/me恢复
export async function getCsrfToken(signal?: AbortSignal): Promise<string> {
  if (!csrfToken) await getCurrentUser(signal)
  return csrfToken
}

// 清除前端认证缓存，不等于服务端退出登录
export function clearAuthState(): void {
  csrfToken = ''
  sessionId = ''
  userId = ''
  tabToken = ''
  try {
    sessionStorage.removeItem(sessionKey)
  } catch {
    /* 无存储权限时只清理内存 */
  }
}

// 后端确认撤销登录凭证后再清除前端缓存；失败时保留当前身份供重试。
// 认证在获取CSRF阶段就返回401时，也应允许本地退出，不能卡在工作台
export async function logout(): Promise<void> {
  // 401处理可能提前清理缓存，发送通知时仍须保留原账号和会话范围。
  const scope = getAuthScope()
  try {
    const response = await authenticatedFetch('/api/auth/logout', { method: 'POST' })
    if (!response.ok && response.status !== 401) {
      throw new Error(await getApiError(response, '退出登录失败，请稍后重试'))
    }
  } catch (error) {
    if (!(error instanceof Error) || error.message !== '登录已失效，请重新登录') throw error
  }
  // 服务器已确认退出或会话已失效后，才通知其它标签页取消请求
  if (sessionId === scope.session_id) notifyAuthExpired()
  const notice = { type: 'logout', ...scope }
  authChannel?.postMessage(notice)
  try {
    localStorage.setItem(
      logoutKey,
      JSON.stringify({ ...notice, nonce: `${Date.now()}-${Math.random()}` }),
    )
  } catch {
    /* 无存储权限时依赖服务端撤销检查 */
  }
}

// 显示后端的实际拒绝原因，区分角色权限、请求来源与CSRF错误
export async function getApiError(response: Response, fallback: string): Promise<string> {
  if (response.status === 401) return '登录已失效，请重新登录'
  const body = await response
    .clone()
    .json()
    .catch(() => ({}))
  return typeof body.detail === 'string' ? body.detail : fallback
}

// Cookie可能被其他标签页的登录更新，缓存CSRF与Cookie需在校验失败时重新同步
export async function authenticatedFetch(
  url: string,
  options: RequestInit = {},
): Promise<Response> {
  const method = (options.method || 'GET').toUpperCase()
  const requiresCsrf = !['GET', 'HEAD', 'OPTIONS'].includes(method)
  const signal = options.signal || undefined
  let requestedSession = sessionId
  const send = async () => {
    const headers = new Headers(options.headers)
    if (requiresCsrf) headers.set('X-CSRF-Token', await getCsrfToken(signal))
    if (!requestedSession) requestedSession = sessionId
    if (requestedSession !== sessionId) throw new Error('请求期间登录已变更，请重试')
    if (!requestedSession || !tabToken) {
      notifyAuthExpired()
      throw new Error('登录已失效，请重新登录')
    }
    headers.set('X-Session-ID', requestedSession)
    headers.set('X-Tab-Token', tabToken)
    return fetch(url, { ...options, headers, credentials: 'same-origin' })
  }
  let response = await send()
  if (requiresCsrf && response.status === 403 && requestedSession === sessionId) {
    const body = await response
      .clone()
      .json()
      .catch(() => ({}))
    // 只有认证依赖在执行业务前拒绝的CSRF错误才能重试，权限和来源错误不重试
    if (body.detail === 'CSRF验证失败') {
      await getCurrentUser(signal)
      response = await send()
    }
  }
  // 迟到的旧会话401不能强制退出当前新登录。
  if (response.status === 401 && requestedSession === sessionId) notifyAuthExpired()
  if (response.ok && requiresCsrf && requestedSession === sessionId) lastActivity = Date.now()
  return response
}

// 只把可见页面的用户操作记为活动；定时检查本身不会延长服务端空闲期。
export function startActivityTracking(): () => void {
  let activityPending = false
  function checkIdle(): boolean {
    if (sessionId && Date.now() - lastActivity >= idleSeconds * 1000) {
      notifyAuthExpired()
      return true
    }
    return false
  }
  const onInteraction = () => {
    if (!sessionId || !tabToken || document.visibilityState !== 'visible' || checkIdle()) return
    lastActivity = Date.now()
    if (activityPending || Date.now() - lastActivityRequest < 30_000) return
    activityPending = true
    lastActivityRequest = Date.now()
    void authenticatedFetch('/api/auth/activity', { method: 'POST' })
      .catch(() => {
        /* 网络故障不伪装成退出，服务端仍独立检查超时 */
      })
      .finally(() => {
        activityPending = false
      })
  }
  const verify = () => {
    if (!sessionId || !tabToken || checkIdle() || document.visibilityState !== 'visible') return
    void getCurrentUser().catch(() => {
      /* 401已触发退出，其余错误等待下次验证 */
    })
  }
  window.addEventListener('pointerdown', onInteraction)
  window.addEventListener('keydown', onInteraction)
  document.addEventListener('visibilitychange', verify)
  const timer = setInterval(verify, 30_000)
  return () => {
    clearInterval(timer)
    window.removeEventListener('pointerdown', onInteraction)
    window.removeEventListener('keydown', onInteraction)
    document.removeEventListener('visibilitychange', verify)
  }
}
