import { authenticatedFetch, getApiError, AUTH_EXPIRED_EVENT, notifyAuthExpired } from './auth'

import { broadcastRunStatus } from './runs'

// 聊天请求结构，user_id暂时保留以兼容后端ChatRequest模型
export interface ChatRequest {
  query: string
  thread_id: string
  user_id: string
}

// SSE事件数据结构
export interface SSEEvent {
  type: string
  data: any
}

// 发送聊天请求并实时接收SSE消息
export async function chatStream(
  request: ChatRequest,
  onMessage: (event: SSEEvent) => void,
): Promise<void> {
  // 从认证模块获取CSRF Token，JWT由浏览器自动携带
  const controller = new AbortController()
  let sessionExpired = false
  const stopForLogout = () => {
    sessionExpired = true
    controller.abort()
  }
  window.addEventListener(AUTH_EXPIRED_EVENT, stopForLogout)
  const timer = setTimeout(() => controller.abort(), 330_000)
  try {
    const response = await authenticatedFetch('/api/chat/stream/sse', {
      signal: controller.signal,
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })

    // 在进入流式读取前处理身份认证和权限错误
    if (response.status === 401) throw new Error('登录已失效，请重新登录')
    if (!response.ok)
      throw new Error(await getApiError(response, `聊天请求失败：HTTP ${response.status}`))
    if (!response.headers.get('content-type')?.toLowerCase().startsWith('text/event-stream')) {
      throw new Error('聊天接口没有返回流式消息，请检查后端服务')
    }
    if (!response.body) throw new Error('SSE连接失败')
    broadcastRunStatus()

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    let completed = false
    function parseEvent(frame: string): void {
      const data = frame
        .split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).replace(/^ /, ''))
        .join('\n')
      if (!data) return
      if (data === '[DONE]') {
        completed = true
        return
      }
      const event = JSON.parse(data)
      if (event.type === 'error') {
        if (event.data?.code === 'session_expired') notifyAuthExpired()
        throw new Error(event.data?.message || '服务端处理失败')
      }
      if (event.type) onMessage(event)
      if (event.type === 'done') completed = true
    }

    function processBuffer(): void {
      let boundary: RegExpExecArray | null
      while (!completed && (boundary = /\r?\n\r?\n/.exec(buffer)) !== null) {
        const frame = buffer.slice(0, boundary.index)
        buffer = buffer.slice(boundary.index + boundary[0].length)
        parseEvent(frame)
      }
    }

    try {
      while (!completed) {
        const { done, value } = await reader.read()
        buffer += done ? decoder.decode() : decoder.decode(value, { stream: true })
        processBuffer()
        if (done) {
          if (!completed) throw new Error('回复连接意外中断，请重试')
          break
        }
      }
    } finally {
      await reader.cancel().catch(() => {})
      reader.releaseLock()
    }
  } catch (error) {
    if (sessionExpired) throw new Error('登录已失效，请重新登录')
    if (controller.signal.aborted) throw new Error('聊天请求超时，请重试')
    throw error
  } finally {
    broadcastRunStatus()
    clearTimeout(timer)
    window.removeEventListener(AUTH_EXPIRED_EVENT, stopForLogout)
  }
}
