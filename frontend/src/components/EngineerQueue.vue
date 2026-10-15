<template>
  <section class="queue-panel">
    <div class="toolbar">
      <strong>工程师待处理问题队列</strong>
      <select v-model="status" :disabled="busy" @change="resetQueue">
        <option value="pending">待接单</option>
        <option value="processing">处理中</option>
        <option value="answered">已回复</option>
        <option value="closed">已关闭</option>
      </select>
      <label
        ><input v-model="mine" type="checkbox" :disabled="busy" @change="resetQueue" />
        仅我的接单</label
      >
      <button :disabled="busy || loading" @click="loadQueue">刷新队列</button>
    </div>
    <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
    <p v-if="loading">正在加载工单...</p>
    <p v-else-if="!questions.length">当前没有{{ statusNames[status] }}的问题。</p>
    <article v-for="question in questions" :key="question.id" class="ticket">
      <div class="ticket-header">
        <strong>工单 #{{ question.id }} · {{ statusNames[question.status] }}</strong>
        <span>{{ formatDate(question.created_at) }}</span>
      </div>
      <p>
        提问人：{{ question.requester_name || '历史知识库问题' }} · 接单人：{{
          question.assigned_name || '未接单'
        }}
      </p>
      <p class="question">{{ question.question }}</p>
      <!-- 证据定位和审核结果只作参考，人工回复由工程师确认 -->
      <details
        v-if="question.evidence_refs?.length || Object.keys(question.verification || {}).length"
      >
        <summary>查看参考来源和审核信息</summary>
        <p v-for="(source, index) in question.evidence_refs" :key="index">
          {{ source.reference }} {{ source.file_name }} {{ source.location }}
        </p>
        <pre>{{ JSON.stringify(question.verification, null, 2) }}</pre>
      </details>
      <button v-if="question.status === 'pending'" :disabled="busy" @click="claim(question.id)">
        接单处理
      </button>
      <!-- 只有当前接单工程师可编辑回复，后端仍会再次校验身份 -->
      <div v-if="question.status === 'processing' && question.assigned_engineer_id === userId">
        <textarea
          v-model="drafts[question.id]"
          maxlength="20000"
          rows="5"
          placeholder="请输入人工核实后的回复，建议说明依据和待确认事项。"
          :disabled="busy"
        ></textarea>
        <button :disabled="busy || !drafts[question.id]?.trim()" @click="reply(question.id)">
          提交人工回复
        </button>
      </div>
      <div v-if="question.engineer_answer" class="answer">
        <strong>人工回复 · {{ formatDate(question.answered_at) }}</strong>
        <p>{{ question.engineer_answer }}</p>
      </div>
    </article>
    <div class="pagination">
      <button :disabled="busy || loading || offset === 0" @click="changePage(-50)">上一页</button>
      <span>第 {{ Math.floor(offset / 50) + 1 }} 页</span>
      <button :disabled="busy || loading || questions.length < 50" @click="changePage(50)">
        下一页
      </button>
    </div>
  </section>
</template>
<script setup lang="ts">
import { onMounted, ref } from 'vue'
import {
  getEngineerQuestions,
  claimQuestion,
  replyQuestion,
  type EngineerQuestion,
  type QuestionStatus,
} from '../api/engineer'

defineProps<{ userId: string }>()
const emit = defineEmits<{ replied: [] }>()
const status = ref<QuestionStatus>('pending')
const mine = ref(false)
const questions = ref<EngineerQuestion[]>([])
const drafts = ref<Record<number, string>>({})
const busy = ref(false)
const loading = ref(false)
const errorMessage = ref('')
const offset = ref(0)
const statusNames = {
  pending: '待接单',
  processing: '处理中',
  answered: '已回复',
  closed: '已关闭',
}
let loadVersion = 0

function formatDate(value: string | null): string {
  if (!value) return ''
  return new Date(value).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' })
}

// 版本号防止快速切换筛选条件时，旧请求覆盖最新队列
async function loadQueue() {
  const version = ++loadVersion
  loading.value = true
  errorMessage.value = ''
  try {
    const result = await getEngineerQuestions(status.value, mine.value, offset.value)
    if (version === loadVersion) questions.value = result
  } catch (error) {
    if (version === loadVersion)
      errorMessage.value = error instanceof Error ? error.message : '加载队列失败'
  } finally {
    if (version === loadVersion) loading.value = false
  }
}

function resetQueue() {
  offset.value = 0
  void loadQueue()
}
function changePage(step: number) {
  offset.value += step
  void loadQueue()
}

async function claim(id: number) {
  busy.value = true
  errorMessage.value = ''
  try {
    await claimQuestion(id)
    status.value = 'processing'
    mine.value = true
    offset.value = 0
    await loadQueue()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '接单失败'
  } finally {
    busy.value = false
  }
}

async function reply(id: number) {
  busy.value = true
  errorMessage.value = ''
  try {
    await replyQuestion(id, drafts.value[id]!.trim())
    delete drafts.value[id]
    await loadQueue()
    emit('replied')
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '回复失败'
  } finally {
    busy.value = false
  }
}

onMounted(loadQueue)
</script>
<style scoped>
.queue-panel {
  width: 100%;
  padding: 24px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  min-height: 450px;
}
.toolbar {
  padding-bottom: 22px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 20px;
}
.toolbar,
.ticket-header,
.pagination {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}
.toolbar strong {
  margin-right: auto;
  font-size: 17px;
}
.toolbar label {
  color: var(--muted);
  font-size: 13px;
}
.ticket {
  padding: 22px;
  border: 1px solid var(--line);
  border-radius: 12px;
  margin-bottom: 16px;
  background: #102237;
}
.ticket-header {
  justify-content: space-between;
  color: var(--cyan);
}
.ticket-header span,
.ticket > p:not(.question) {
  font-size: 12px;
  color: var(--muted);
}
.question,
.answer p {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  line-height: 1.8;
}
.answer {
  padding: 16px;
  background: #103135;
  border: 1px solid #27514f;
  border-radius: 9px;
  margin-top: 16px;
}
.answer strong {
  color: #91e2d0;
  font-size: 13px;
}
textarea {
  width: 100%;
  margin: 16px 0;
  padding: 14px;
  border: 1px solid #2b4b66;
  border-radius: 9px;
  background: #091828;
  color: var(--text);
  resize: vertical;
}
button,
select {
  padding: 10px 14px;
  border: 1px solid #2b4b66;
  border-radius: 8px;
  background: #122d45;
  color: #b4d9ed;
}
.ticket > button,
.ticket textarea + button {
  background: #2867d8;
  color: white;
  border-color: #367eec;
}
.error {
  color: #ffadad;
  padding: 12px;
  border-radius: 8px;
  background: #321d29;
}
details {
  color: var(--muted);
  font-size: 13px;
  margin: 16px 0;
}
summary {
  cursor: pointer;
  color: #9acde5;
}
pre {
  overflow: auto;
  max-height: 240px;
  background: #081625;
  padding: 14px;
  border-radius: 8px;
}
.pagination {
  margin-top: 24px;
  justify-content: flex-end;
  color: var(--muted);
  font-size: 13px;
}
@media (max-width: 600px) {
  .queue-panel {
    padding: 14px;
  }
  .ticket {
    padding: 14px;
  }
  .toolbar strong {
    width: 100%;
  }
}
</style>
