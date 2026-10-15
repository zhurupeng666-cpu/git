<template>
  <div v-if="loading" class="loading"><span class="loading-orbit"></span>正在验证登录状态...</div>
  <LoginView v-else-if="!user" @logged-in="handleLoggedIn" />
  <div v-else class="workspace">
    <!-- 全局品牌和身份信息，在不同工作页面保持一致 -->
    <header class="header">
      <div class="brand">
        <span class="brand-icon" aria-hidden="true">◈</span>
        <div><strong>半导体智能知识库</strong><small>SEMICONDUCTOR INTELLIGENCE</small></div>
      </div>
      <div class="user-info">
        <span class="avatar" aria-hidden="true">{{ user.display_name.slice(0, 1) }}</span>
        <span>{{ user.display_name }}</span>
        <span class="role">{{ user.role === 'engineer' ? '工程师' : '普通员工' }}</span>
        <button class="logout-btn" :disabled="loggingOut" @click="handleLogout">
          {{ loggingOut ? '正在退出...' : '退出登录' }}
        </button>
      </div>
    </header>
    <!-- 导航栏将会话和工程师队列分离，工程师入口仅向工程师显示 -->
    <nav class="navigation" aria-label="工作台导航">
      <RouterLink to="/chat" class="nav-link"
        ><span aria-hidden="true">◇</span> 知识库问答</RouterLink
      >
      <RouterLink v-if="user.role === 'engineer'" to="/engineer" class="nav-link"
        ><span aria-hidden="true">▤</span> 工程师工作台</RouterLink
      >
      <span class="nav-caption">{{
        user.role === 'engineer' ? '工程师协作空间' : '员工知识服务空间'
      }}</span>
    </nav>
    <main class="main">
      <div class="page-heading">
        <div>
          <p class="eyebrow">
            {{ route.name === 'engineer' ? 'HUMAN EXPERTISE' : 'KNOWLEDGE EXPLORER' }}
          </p>
          <h1>{{ route.name === 'engineer' ? '人工处理中心' : '知识探索与智能问答' }}</h1>
          <p class="page-description">
            {{
              route.name === 'engineer'
                ? '领取待处理问题，核实知识依据，让专业经验形成可靠答案。'
                : '连接工艺知识与材料数据，从一个问题开始探索。'
            }}
          </p>
        </div>
        <span class="page-tag">{{
          route.name === 'engineer' ? '人工核实 · 专业协作' : '知识检索 · 数据查询 · 报告生成'
        }}</span>
      </div>
      <div v-if="logoutError" class="error">{{ logoutError }}</div>
      <!-- 仅缓存聊天页，切换导航时保留当前会话与流式请求 -->
      <RouterView v-slot="{ Component, route: currentRoute }">
        <KeepAlive include="ChatBox">
          <component
            :is="Component"
            v-bind="currentRoute.name === 'engineer' ? { userId: user.user_id } : {}"
          />
        </KeepAlive>
      </RouterView>
    </main>
  </div>
</template>
<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { RouterLink, RouterView, useRoute, useRouter } from 'vue-router'
import LoginView from './views/LoginView.vue'
import {
  getCurrentUser,
  logout,
  startActivityTracking,
  AUTH_EXPIRED_EVENT,
  type UserProfile,
} from './api/auth'

const user = ref<UserProfile | null>(null)
const loading = ref(true)
const loggingOut = ref(false)
const logoutError = ref('')
// 其它标签页退出或服务端撤销后，同步返回登录页并卸载聊天组件
function handleAuthExpired() {
  user.value = null
  logoutError.value = ''
}
window.addEventListener(AUTH_EXPIRED_EVENT, handleAuthExpired)
onUnmounted(() => window.removeEventListener(AUTH_EXPIRED_EVENT, handleAuthExpired))
const route = useRoute()
const router = useRouter()

// 路由入口进行界面权限校验，后端接口仍独立校验工程师角色
watch([() => route.name, () => user.value?.role, loading], () => {
  if (!loading.value && route.name === 'engineer' && user.value?.role !== 'engineer') {
    void router.replace('/chat')
  }
})

let stopActivityTracking: (() => void) | undefined
onUnmounted(() => stopActivityTracking?.())
onMounted(async () => {
  stopActivityTracking = startActivityTracking()
  try {
    user.value = await getCurrentUser()
  } catch {
    user.value = null
  } finally {
    loading.value = false
  }
})

function handleLoggedIn(profile: UserProfile) {
  user.value = profile
  logoutError.value = ''
}

// 服务端撤销登录成功后再返回登录页，失败时保留当前工作状态
async function handleLogout() {
  if (loggingOut.value) return
  loggingOut.value = true
  logoutError.value = ''
  try {
    await logout()
    user.value = null
  } catch (error) {
    logoutError.value = error instanceof Error ? error.message : '退出登录失败'
  } finally {
    loggingOut.value = false
  }
}
</script>
<style scoped>
.workspace {
  min-height: 100vh;
}
.header {
  min-height: 78px;
  padding: 0 4%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  border-bottom: 1px solid var(--line);
  background: rgba(9, 20, 36, 0.9);
}
.brand,
.user-info {
  display: flex;
  align-items: center;
  gap: 14px;
}
.brand strong {
  font-size: 19px;
  letter-spacing: 1px;
}
.brand small {
  display: block;
  margin-top: 5px;
  color: var(--muted);
  font-size: 10px;
  letter-spacing: 2px;
}
.brand-icon {
  display: grid;
  place-items: center;
  width: 44px;
  height: 44px;
  color: var(--cyan);
  background: #102a43;
  border: 1px solid #245675;
  border-radius: 12px;
  font-size: 30px;
}
.user-info {
  font-size: 13px;
}
.avatar {
  width: 32px;
  height: 32px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: #193653;
  color: #a4dcff;
}
.role {
  padding: 5px 10px;
  border-radius: 6px;
  background: #12304b;
  color: #8cd8fa;
}
.logout-btn {
  background: transparent;
  color: var(--muted);
  padding: 8px 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
}
.navigation {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 4%;
  border-bottom: 1px solid var(--line);
}
.nav-link {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 11px 18px;
  border-radius: 8px;
  text-decoration: none;
  color: var(--muted);
  border: 1px solid transparent;
}
.nav-link.router-link-active {
  color: #b5eafa;
  background: #123047;
  border-color: #23516b;
}
.nav-caption {
  margin-left: auto;
  font-size: 12px;
  color: var(--muted);
}
.main {
  padding: 24px 4% 32px;
}
.page-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 24px;
}
.eyebrow {
  color: var(--cyan);
  letter-spacing: 2px;
  font-size: 10px;
  margin: 0 0 8px;
}
h1 {
  margin: 0;
  font-size: 26px;
  letter-spacing: 1px;
}
.page-description {
  margin: 10px 0 0;
  color: var(--muted);
  font-size: 13px;
}
.page-tag {
  padding: 9px 12px;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 20px;
  font-size: 11px;
}
.error {
  padding: 12px;
  color: #ffadad;
  background: #321d29;
  border-radius: 8px;
  margin-bottom: 16px;
}
.loading {
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 16px;
  color: var(--muted);
}
.loading-orbit {
  width: 24px;
  height: 24px;
  border: 2px solid #1c3b55;
  border-top-color: var(--cyan);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
@media (max-width: 700px) {
  .header {
    padding: 16px;
    flex-wrap: wrap;
  }
  .brand strong {
    font-size: 16px;
  }
  .user-info {
    gap: 8px;
  }
  .navigation {
    padding: 10px 16px;
  }
  .nav-caption,
  .page-tag {
    display: none;
  }
  .nav-link {
    padding: 10px;
    font-size: 13px;
  }
  .main {
    padding: 20px 12px;
  }
  h1 {
    font-size: 22px;
  }
}
@media (prefers-reduced-motion: reduce) {
  .loading-orbit {
    animation: none;
  }
}
</style>
