import { authenticatedFetch, getApiError } from './auth'

// RAG引用来源
export interface ChatSource {
  file_name: string
  domain?: string
  title?: string
  page?: number
  chunk_id?: string
  location?: string
  regions?: number[][]
  parse_method?: string
  quality_status?: string
}

// 会话信息
export interface ChatSession {
  id: string
  user_id?: string
  title: string
  created_at: string
  updated_at: string
}

// 历史聊天消息
export interface ChatMessage {
  id: string
  sender_role: 'user' | 'assistant' | 'engineer'
  sender_user_id: string | null
  content: string
  sources: ChatSource[]
  created_at: string
}

// 统一处理API错误
async function handleResponse<T>(response: Response): Promise<T> {
  if (response.status === 401) throw new Error('登录已失效，请重新登录')
  if (response.status === 404) throw new Error('会话不存在或无权访问')
  if (!response.ok)
    throw new Error(await getApiError(response, `请求失败：HTTP ${response.status}`))
  return response.json() as Promise<T>
}

// 创建新会话，后端根据JWT自动确定所属用户
export async function createSession(title = '新会话'): Promise<ChatSession> {
  const response = await authenticatedFetch('/api/sessions', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
  return handleResponse<ChatSession>(response)
}

// 获取当前用户的会话列表，支持分页
export async function getSessions(limit = 50, offset = 0): Promise<ChatSession[]> {
  const response = await authenticatedFetch(`/api/sessions?limit=${limit}&offset=${offset}`, {
    credentials: 'same-origin',
  })
  const result = await handleResponse<{ sessions: ChatSession[] }>(response)
  return result.sessions
}

// 获取指定会话的历史聊天消息
export async function getSessionMessages(
  sessionId: string,
  limit = 100,
  offset = 0,
): Promise<ChatMessage[]> {
  const response = await authenticatedFetch(
    `/api/sessions/${encodeURIComponent(sessionId)}/messages?limit=${limit}&offset=${offset}`,
    {
      credentials: 'same-origin',
    },
  )
  const result = await handleResponse<{ messages: ChatMessage[] }>(response)
  return result.messages
}

// 删除会话，后端执行软删除并校验会话归属
export async function deleteSession(sessionId: string): Promise<void> {
  const response = await authenticatedFetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
    credentials: 'same-origin',
  })
  await handleResponse<{ message: string }>(response)
}
