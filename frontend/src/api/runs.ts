import { authenticatedFetch, getApiError, getAuthScope } from './auth'

export const RUN_STATUS_EVENT = 'semiconductor-run-status'
const channel =
  typeof window.BroadcastChannel === 'function'
    ? new window.BroadcastChannel('semiconductor-runs')
    : null
let version = 0

// 通知只触发重新查询，不能用另一个标签页的消息代替后端任务状态。
export async function refreshRunStatus(): Promise<void> {
  const current = ++version
  try {
    const response = await authenticatedFetch('/api/chat/run')
    if (!response.ok) throw new Error(await getApiError(response, '获取任务状态失败'))
    const data = await response.json()
    if (current === version)
      window.dispatchEvent(new CustomEvent(RUN_STATUS_EVENT, { detail: Boolean(data.running) }))
  } catch {
    // 请求失败时保留最后状态，发送接口仍会执行数据库并发校验。
  }
}

export function broadcastRunStatus(): void {
  channel?.postMessage({ type: 'changed', ...getAuthScope() })
  void refreshRunStatus()
}

if (channel)
  channel.onmessage = (event) => {
    const scope = getAuthScope()
    // 其他账号的任务变化不能触发本页面的认证请求或忙碌状态变化。
    if (
      scope.user_id &&
      event.data?.type === 'changed' &&
      event.data.user_id === scope.user_id &&
      event.data.session_id === scope.session_id
    )
      void refreshRunStatus()
  }
