import { authenticatedFetch, getApiError } from './auth'

// 人工工单信息，兼容旧版没有会话关联的待答问题
export type QuestionStatus = 'pending' | 'processing' | 'answered' | 'closed'
export interface EngineerQuestion {
  id: number
  question: string
  status: QuestionStatus
  session_id: string | null
  requester_name?: string | null
  assigned_engineer_id: string | null
  assigned_name?: string | null
  engineer_answer: string | null
  verification: Record<string, unknown>
  evidence_refs: { file_name?: string; location?: string; reference?: string }[]
  created_at: string
  answered_at: string | null
}

// GET自动携带登录Cookie，POST额外携带CSRF Token
async function request<T>(url: string, body?: object): Promise<T> {
  const headers: Record<string, string> = {}
  if (body) {
    headers['Content-Type'] = 'application/json'
  }
  const response = await authenticatedFetch(url, {
    method: body ? 'POST' : 'GET',
    credentials: 'same-origin',
    headers,
    ...(body ? { body: JSON.stringify(body) } : {}),
  })
  if (!response.ok) {
    throw new Error(await getApiError(response, `工单请求失败：HTTP ${response.status}`))
  }
  return response.json() as Promise<T>
}

export async function getEngineerQuestions(status: QuestionStatus, mine = false, offset = 0) {
  const result = await request<{ questions: EngineerQuestion[] }>(
    `/api/engineer/questions?status=${status}&mine=${mine}&limit=50&offset=${offset}`,
  )
  return result.questions
}

export function claimQuestion(id: number) {
  return request<EngineerQuestion>(`/api/engineer/questions/${id}/claim`, {})
}

export function replyQuestion(id: number, answer: string) {
  return request<EngineerQuestion>(`/api/engineer/questions/${id}/reply`, { answer })
}

export function submitEngineerQuestion(sessionId: string, question: string) {
  return request<EngineerQuestion>(
    `/api/sessions/${encodeURIComponent(sessionId)}/engineer-questions`,
    { question },
  )
}

export async function getSessionQuestions(sessionId: string) {
  const result = await request<{ questions: EngineerQuestion[] }>(
    `/api/sessions/${encodeURIComponent(sessionId)}/engineer-questions`,
  )
  return result.questions
}
