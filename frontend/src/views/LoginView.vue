<template>
  <div class="login-page">
    <div class="login-card">
      <!-- 系统标题 -->
      <div class="logo">◈</div>
      <h1>半导体智能知识库</h1>
      <p class="subtitle">Semiconductor Knowledge Assistant</p>
      <!-- 用户登录表单 -->
      <form @submit.prevent="handleLogin">
        <label for="username">用户名</label>
        <input
          id="username"
          v-model.trim="username"
          autocomplete="username"
          placeholder="请输入员工账号"
          required
        />
        <label for="password">密码</label>
        <input
          id="password"
          v-model="password"
          type="password"
          autocomplete="current-password"
          placeholder="请输入登录密码"
          required
        />
        <!-- 登录失败提示 -->
        <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
        <button type="submit" :disabled="loading">
          {{ loading ? '正在验证身份...' : '登录系统' }}
        </button>
      </form>
      <p class="footer">企业内部知识库 · 安全身份认证</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { login, type UserProfile } from '../api/auth'

// 登录成功后将用户信息交给父组件处理
const emit = defineEmits<{ loggedIn: [user: UserProfile] }>()
const username = ref('')
const password = ref('')
const loading = ref(false)
const errorMessage = ref('')

// 调用FastAPI认证接口，JWT由后端写入HttpOnly Cookie
async function handleLogin() {
  if (loading.value || !username.value || !password.value) return
  loading.value = true
  errorMessage.value = ''
  try {
    const user = await login(username.value, password.value)
    password.value = ''
    // 用户角色由后端返回，不能由前端自行指定
    emit('loggedIn', user)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '登录失败，请稍后重试'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: radial-gradient(ellipse at top, #153550, var(--bg) 65%);
  padding: 20px;
  box-sizing: border-box;
}
.login-card {
  width: 100%;
  max-width: 400px;
  padding: 44px 40px;
  background: var(--panel);
  border: 1px solid #20364c;
  border-radius: 14px;
  box-shadow: 0 12px 40px #1c35540d;
  box-sizing: border-box;
}
.logo {
  color: #2563eb;
  font-size: 38px;
  text-align: center;
  margin-bottom: 10px;
}
h1 {
  font-size: 24px;
  color: var(--text);
  text-align: center;
  margin: 0;
}
.subtitle {
  color: var(--muted);
  font-size: 13px;
  text-align: center;
  margin: 10px 0 35px;
}
form {
  display: flex;
  flex-direction: column;
}
label {
  color: var(--muted);
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 8px;
}
input {
  color: var(--text);
  background: #081626;
  height: 44px;
  border: 1px solid #dbe2ea;
  border-radius: 7px;
  padding: 0 13px;
  margin-bottom: 20px;
  font-size: 14px;
  outline: none;
  box-sizing: border-box;
}
input:focus {
  border-color: #2563eb;
}
button {
  height: 44px;
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 7px;
  font-size: 15px;
  cursor: pointer;
  margin-top: 5px;
}
button:hover:not(:disabled) {
  background: #1d4ed8;
}
button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.error {
  color: #dc2626;
  font-size: 13px;
  margin: 0 0 12px;
}
.footer {
  text-align: center;
  color: var(--muted);
  font-size: 12px;
  margin: 30px 0 0;
}
@media (max-width: 480px) {
  .login-card {
    padding: 30px 24px;
  }
}
</style>
