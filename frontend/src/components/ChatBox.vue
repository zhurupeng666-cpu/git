<template>
  <div class="chat-layout">
    <!-- 左侧会话管理 -->
    <aside class="sidebar">
      <button
        class="new-btn"
        :disabled="sending || loading || transferBusy || refreshing"
        @click="newSession"
      >
        + 新建会话
      </button>
      <div class="session-title">历史会话</div>
      <div
        v-for="session in sessions"
        :key="session.id"
        class="session-row"
        :class="{ active: activeSessionId === session.id }"
      >
        <button
          class="session-name"
          :disabled="sending || transferBusy || refreshing"
          @click="selectSession(session.id)"
          :title="`${session.title === '新会话' ? '未提问' : session.title} · ${formatSessionDate(session.created_at)}`"
        >
          <span class="session-question">{{
            session.title === '新会话' ? '未提问' : session.title
          }}</span>
          <time class="session-date" :datetime="session.created_at">{{
            formatSessionDate(session.created_at)
          }}</time>
        </button>
        <button
          class="delete-btn"
          :disabled="sending || transferBusy || refreshing"
          title="删除会话"
          @click="removeSession(session.id)"
        >
          ×
        </button>
      </div>
      <p v-if="!sessions.length && !loading" class="empty">暂无历史会话</p>
    </aside>
    <!-- 右侧聊天区域 -->
    <section class="chat-main">
      <div ref="messageContainer" class="messages">
        <p v-if="loading" class="empty">正在加载历史会话...</p>
        <section v-else-if="!messages.length" class="welcome">
          <h2>欢迎使用半导体智能知识库</h2>
          <p>你可以直接提问，也可以让我查询材料数据或生成报告：</p>
          <ul>
            <li>
              <strong>知识库问答</strong
              >：工艺、设备维护、故障排查和规范，例如“CVD腔体压力异常如何排查？”
            </li>
            <li><strong>当前库存查询</strong>：按物料编码查看库存、仓库和批次。</li>
            <li><strong>出入库流水</strong>：按物料编码和日期范围查询记录。</li>
            <li>
              <strong>周期库存统计</strong>：查看期初库存、入库、出库和期末库存，可限定分类或物料。
            </li>
            <li><strong>周报/月报</strong>：生成全部材料、某类材料或指定物料的管理报告。</li>
            <li>
              <strong>人工咨询</strong
              >：问题旁点击“转人工处理”，工程师接单后回复，可刷新查看处理状态与人工回复。
            </li>
          </ul>
          <p>
            知识库问题可以直接提出；数据查询和报告缺少必要信息时，我会先追问。左侧支持新建、查看和删除历史会话。
          </p>
        </section>
        <!-- 历史消息与实时消息统一展示 -->
        <article v-for="(msg, index) in messages" :key="index" class="message">
          <div class="role">{{ msg.role }}</div>
          <!-- 每个用户问题都可转人工，工单状态随当前会话一起加载 -->
          <div v-if="msg.role === '用户'" class="human-actions">
            <span v-if="questionTicket(msg.content)"
              >工单 #{{ questionTicket(msg.content)!.id }} ·
              {{ ticketStatusNames[questionTicket(msg.content)!.status] }}</span
            >
            <button
              v-else
              :disabled="sending || transferBusy"
              @click="transferQuestion(msg.content)"
            >
              转人工处理
            </button>
          </div>
          <div v-if="msg.status" class="status">{{ msg.status }}</div>
          <!-- 使用Markdown渲染回答，DOMPurify过滤不安全HTML -->
          <div v-if="msg.content" class="content" v-html="renderMarkdown(msg.content)"></div>
          <!-- 展示RAG来源 -->
          <div v-if="msg.sources.length" class="sources">
            <div class="source-title">📄 来源文档</div>
            <div v-for="(source, i) in msg.sources" :key="i" class="source-item">
              {{ source.file_name }}
              <!-- 扫描资料保留原始页码和区域，便于回到原文复核。 -->
              <span v-if="source.location"> · {{ source.location }}</span>
              <span v-else-if="source.page"> · 第{{ source.page }}页</span>
            </div>
          </div>
        </article>
      </div>
      <div class="human-actions">
        <button :disabled="sending || refreshing || !activeSessionId" @click="refreshActiveSession">
          {{ refreshing ? '正在刷新...' : '刷新人工回复与工单状态' }}
        </button>
        <span v-if="notice">{{ notice }}</span>
      </div>
      <div v-if="accountRunning && !sending" class="status">
        当前账号正在其他页面处理问题，请等待完成后再发送。
      </div>
      <!-- API异常提示 -->
      <div v-if="errorMessage" class="error">{{ errorMessage }}</div>
      <!-- 用户输入区域 -->
      <form class="input-area" @submit.prevent="send">
        <input
          v-model="query"
          :disabled="sending || loading"
          placeholder="请输入知识库问题、材料查询或报告需求..."
        />
        <button type="submit" :disabled="sending || loading || accountRunning || !query.trim()">
          {{ sending ? '生成中...' : '发送' }}
        </button>
      </form>
    </section>
  </div>
</template>
<script setup lang="ts">
import { nextTick, onActivated, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import MarkdownIt from 'markdown-it'
import DOMPurify from 'dompurify'
import { chatStream } from '../api/chat'
import { RUN_STATUS_EVENT, refreshRunStatus } from '../api/runs'
import { submitEngineerQuestion, getSessionQuestions, type EngineerQuestion } from '../api/engineer'
import {
  createSession,
  getSessions,
  getSessionMessages,
  deleteSession,
  type ChatSession,
  type ChatSource,
} from '../api/sessions'

// 页面消息结构
interface Message {
  role: string
  content: string
  status: string
  sources: ChatSource[]
}

// 初始化聊天状态
const markdown = new MarkdownIt({ html: false, breaks: true, linkify: true })
const sessions = ref<ChatSession[]>([])
const activeSessionId = ref('')
const messages = ref<Message[]>([])
const query = ref('')
const sending = ref(false)
const accountRunning = ref(false)
let runTimer: ReturnType<typeof setInterval> | undefined
const updateRunStatus = (event: Event) => {
  accountRunning.value = Boolean((event as CustomEvent<boolean>).detail)
}
// 页面通知加速同步，定时复查补足新标签页和跨设备任务变化。
onMounted(() => {
  window.addEventListener(RUN_STATUS_EVENT, updateRunStatus)
  void refreshRunStatus()
  runTimer = setInterval(() => void refreshRunStatus(), 2000)
})
onUnmounted(() => {
  window.removeEventListener(RUN_STATUS_EVENT, updateRunStatus)
  clearInterval(runTimer)
})
const loading = ref(true)
const errorMessage = ref('')
const messageContainer = ref<HTMLElement | null>(null)
let loadVersion = 0
const tickets = ref<EngineerQuestion[]>([])
const transferBusy = ref(false)
const refreshing = ref(false)
const notice = ref('')
const ticketStatusNames = {
  pending: '待接单',
  processing: '处理中',
  answered: '已回复',
  closed: '已关闭',
}

function questionTicket(question: string) {
  return tickets.value.find((item) => item.question === question)
}

// 转人工请求只提交当前会话问题，用户身份由后端Cookie确定
async function transferQuestion(question: string) {
  if (!activeSessionId.value || transferBusy.value || sending.value) return
  transferBusy.value = true
  errorMessage.value = ''
  try {
    const ticket = await submitEngineerQuestion(activeSessionId.value, question)
    tickets.value.unshift(ticket)
    notice.value = `已提交工单 #${ticket.id}，等待工程师接单。`
  } catch (error) {
    errorMessage.value = getError(error)
  } finally {
    transferBusy.value = false
  }
}

// 用户主动刷新人工回复，避免在输入问题或切换会话时被后台轮询覆盖
async function refreshActiveSession() {
  if (!activeSessionId.value || sending.value || transferBusy.value || refreshing.value) return
  refreshing.value = true
  try {
    await selectSession(activeSessionId.value)
  } finally {
    refreshing.value = false
  }
}
defineExpose({ refresh: refreshActiveSession })

// 聊天页被缓存时，返回页面再读取人工回复，不打断正在生成的答案
let activatedOnce = false
onActivated(() => {
  if (activatedOnce) void refreshActiveSession()
  activatedOnce = true
})

// 日期始终按北京时间显示，不受浏览器所在时区影响。
function formatSessionDate(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return new Intl.DateTimeFormat('sv-SE', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).format(date)
}

// 安全渲染Markdown
function renderMarkdown(content: string): string {
  return DOMPurify.sanitize(markdown.render(content))
}

// 将聊天区域滚动到最新消息
async function scrollToBottom() {
  await nextTick()
  if (messageContainer.value) messageContainer.value.scrollTop = messageContainer.value.scrollHeight
}
watch(messages, scrollToBottom, { deep: true })

// 初始化当前用户的会话列表
onMounted(async () => {
  try {
    sessions.value = await getSessions()
    if (sessions.value.length) await selectSession(sessions.value[0]!.id)
  } catch (error) {
    errorMessage.value = getError(error)
  } finally {
    loading.value = false
  }
})

// 统一获取错误信息
function getError(error: unknown): string {
  return error instanceof Error ? error.message : '操作失败，请稍后重试'
}

// 加载指定会话的历史消息
async function selectSession(sessionId: string) {
  if (sending.value || transferBusy.value) return
  const version = ++loadVersion
  activeSessionId.value = sessionId
  messages.value = []
  tickets.value = []
  notice.value = ''
  errorMessage.value = ''
  try {
    const [history, questions] = await Promise.all([
      getSessionMessages(sessionId),
      getSessionQuestions(sessionId),
    ])
    if (version !== loadVersion) return
    tickets.value = questions
    messages.value = history.map((item) => ({
      role:
        item.sender_role === 'user' ? '用户' : item.sender_role === 'engineer' ? '工程师' : 'AI',
      content: item.content,
      status: '',
      sources: Array.isArray(item.sources) ? item.sources : [],
    }))
  } catch (error) {
    if (version === loadVersion) errorMessage.value = getError(error)
  }
}

// 创建数据库会话并设为当前会话
async function createAndActivateSession(title = '新会话'): Promise<string> {
  const session = await createSession(title)
  ++loadVersion
  sessions.value.unshift(session)
  activeSessionId.value = session.id
  messages.value = []
  tickets.value = []
  notice.value = ''
  return session.id
}

// 点击新建会话
async function newSession() {
  if (sending.value || transferBusy.value) return
  errorMessage.value = ''
  try {
    await createAndActivateSession()
  } catch (error) {
    errorMessage.value = getError(error)
  }
}

// 删除当前用户的会话
async function removeSession(sessionId: string) {
  if (
    sending.value ||
    transferBusy.value ||
    refreshing.value ||
    !window.confirm('确定删除这个会话吗？')
  )
    return
  errorMessage.value = ''
  try {
    await deleteSession(sessionId)
    sessions.value = sessions.value.filter((item) => item.id !== sessionId)
    if (activeSessionId.value === sessionId) {
      ++loadVersion
      activeSessionId.value = ''
      messages.value = []
      if (sessions.value.length) await selectSession(sessions.value[0]!.id)
    }
  } catch (error) {
    errorMessage.value = getError(error)
  }
}

// 发送问题，使用真实会话UUID执行LangGraph流式问答
async function send() {
  const question = query.value.trim()
  if (!question || sending.value || loading.value || accountRunning.value) return
  sending.value = true
  errorMessage.value = ''
  const aiMessage = reactive<Message>({
    role: 'AI',
    content: '',
    status: '正在处理你的请求...',
    sources: [],
  })
  try {
    // 第一次提问时自动创建会话
    const threadId =
      activeSessionId.value ||
      (await createAndActivateSession(question.replace(/\s+/g, ' ').slice(0, 200)))
    const session = sessions.value.find((item) => item.id === threadId)
    if (session?.title === '新会话') session.title = question.replace(/\s+/g, ' ').slice(0, 200)
    messages.value.push({ role: '用户', content: question, status: '', sources: [] })
    query.value = ''

    messages.value.push(aiMessage)
    // user_id仅用于兼容旧请求模型，后端使用JWT确定真实用户身份
    await chatStream({ query: question, thread_id: threadId, user_id: '' }, (event) => {
      if (event.type === 'status') aiMessage.status = event.data?.message || ''
      if (event.type === 'token') {
        aiMessage.status = ''
        aiMessage.content += event.data?.content || ''
      }
      if (event.type === 'source')
        aiMessage.sources = Array.isArray(event.data) ? event.data : event.data?.sources || []
      if (event.type === 'done') aiMessage.status = ''
    })
    // 更新会话排序，数据库已在回答完成后更新updated_at
    sessions.value = await getSessions()
  } catch (error) {
    errorMessage.value = getError(error)
    aiMessage.status = '请求失败，请查看下方提示'
    if (!aiMessage.content && !query.value) query.value = question
  } finally {
    sending.value = false
  }
}
</script>
<style scoped>
.chat-layout {
  width: 100%;
  height: calc(100dvh - 285px);
  min-height: 480px;
  display: flex;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 20px 50px #0002;
}
.sidebar {
  width: 240px;
  flex-shrink: 0;
  padding: 20px 14px;
  background: #0a192a;
  border-right: 1px solid var(--line);
  overflow-y: auto;
}
.new-btn {
  width: 100%;
  padding: 13px;
  background: linear-gradient(120deg, #2865d6, #357ef0);
  border: 1px solid #5195ff66;
  border-radius: 9px;
  color: white;
  box-shadow: 0 6px 20px #225cca25;
}
.session-title {
  margin: 28px 8px 12px;
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 2px;
}
.session-row {
  display: flex;
  align-items: center;
  border: 1px solid transparent;
  border-radius: 9px;
  margin-bottom: 8px;
}
.session-row.active {
  background: #15314a;
  border-color: #285376;
}
.session-name {
  flex: 1;
  min-width: 0;
  padding: 12px 9px;
  background: transparent;
  color: var(--text);
  border: 0;
  text-align: left;
}
.session-question {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13px;
}
.session-date {
  display: block;
  margin-top: 6px;
  color: var(--muted);
  font-size: 11px;
}
.delete-btn {
  padding: 5px 10px;
  border: 0;
  background: transparent;
  color: #718aa4;
  font-size: 20px;
}
.delete-btn:hover {
  color: #ff9baf;
}
.chat-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}
.messages {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 28px;
}
.message {
  margin-bottom: 22px;
  padding: 18px 20px;
  border: 1px solid var(--line);
  background: #102237;
  border-radius: 12px;
}
.role {
  font-weight: 600;
  color: var(--cyan);
  margin-bottom: 10px;
  font-size: 12px;
  letter-spacing: 1px;
}
.status {
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 8px;
}
.content {
  line-height: 1.85;
  overflow-wrap: anywhere;
  font-size: 14px;
}
.content :deep(p) {
  margin: 0 0 10px;
}
.content :deep(pre) {
  padding: 14px;
  background: #071525;
  border-radius: 8px;
  overflow-x: auto;
}
.content :deep(table) {
  border-collapse: collapse;
  max-width: 100%;
  display: block;
  overflow-x: auto;
}
.content :deep(td),
.content :deep(th) {
  padding: 8px 12px;
  border: 1px solid var(--line);
}
.sources {
  margin-top: 14px;
  padding: 14px;
  background: #0b1c2d;
  border: 1px solid var(--line);
  border-radius: 8px;
}
.source-title {
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--cyan);
}
.source-item {
  font-size: 12px;
  color: var(--muted);
  margin: 5px 0;
}
.input-area {
  display: flex;
  gap: 12px;
  padding: 18px;
  border-top: 1px solid var(--line);
  background: #0b192b;
}
.input-area input {
  flex: 1;
  min-width: 0;
  padding: 14px 16px;
  color: var(--text);
  background: #081626;
  border: 1px solid #2a4660;
  border-radius: 9px;
}
.input-area button {
  width: 90px;
  border: 0;
  border-radius: 9px;
  background: #3376ee;
  color: white;
}
.human-actions {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
  padding: 8px 20px;
  color: var(--muted);
  font-size: 12px;
}
.human-actions button {
  padding: 6px 10px;
  border: 1px solid #294761;
  border-radius: 6px;
  color: #a2cce5;
  background: #10263a;
}
.error {
  padding: 12px 18px;
  color: #ffadad;
  background: #321d29;
  font-size: 13px;
}
.empty {
  color: var(--muted);
  text-align: center;
  margin-top: 40px;
}
.welcome {
  max-width: 850px;
  margin: 24px auto;
  color: var(--muted);
  line-height: 1.8;
}
.welcome h2 {
  margin-bottom: 8px;
  font-size: 24px;
  color: var(--text);
}
.welcome ul {
  list-style: none;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin: 24px 0;
}
.welcome li {
  padding: 18px;
  background: #10263b;
  border: 1px solid #233e55;
  border-radius: 10px;
  font-size: 13px;
}
.welcome strong {
  display: block;
  color: #a4e8f5;
  margin-bottom: 6px;
  font-size: 15px;
}
@media (max-width: 900px) {
  .sidebar {
    width: 190px;
  }
  .messages {
    padding: 18px;
  }
  .welcome ul {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 600px) {
  .chat-layout {
    flex-direction: column;
    height: calc(100dvh - 320px);
    min-height: 550px;
  }
  .sidebar {
    width: 100%;
    max-height: 160px;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
  .session-title {
    margin: 14px 0 8px;
  }
  .new-btn {
    padding: 9px;
  }
  .messages {
    padding: 12px;
  }
  .welcome {
    margin: 12px 0;
  }
  .welcome h2 {
    font-size: 20px;
  }
  .input-area {
    padding: 12px;
  }
}
</style>
