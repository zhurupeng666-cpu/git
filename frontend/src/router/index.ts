import { createRouter, createWebHistory } from 'vue-router'
import ChatBox from '../components/ChatBox.vue'
import EngineerQueue from '../components/EngineerQueue.vue'

// 知识库和人工处理使用独立路由，刷新或直接打开地址仍能恢复当前页面
const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/chat' },
    { path: '/chat', name: 'chat', component: ChatBox },
    { path: '/engineer', name: 'engineer', component: EngineerQueue },
    { path: '/:pathMatch(.*)*', redirect: '/chat' },
  ],
})

export default router
