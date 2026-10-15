// 使用两个独立页面作用域验证通知、SSE取消和409提示，不需要真实登录密码。
const fs = require('node:fs')
const path = require('node:path')
const Module = require('node:module')
const assert = require('node:assert/strict')
const root = path.resolve(__dirname, '..')
const ts = require(path.join(root, 'node_modules/typescript'))
const channels = []
class TestChannel {
  constructor(name) { this.name = name; channels.push(this) }
  postMessage(data) {
    for (const channel of channels)
      if (channel !== this && channel.name === this.name)
        queueMicrotask(() => channel.onmessage?.({ data }))
  }
}
function page() {
  const window = new EventTarget()
  window.BroadcastChannel = TestChannel
  const modules = {}
  function load(name) {
    if (modules[name]) return modules[name]
    const file = path.join(root, 'src/api', name + '.ts')
    const mod = new Module(file)
    mod.testWindow = window
    mod.testSessionStorage = { getItem() { return JSON.stringify({ session_id: 'synthetic-session', tab_token: 'test-tab' }) }, setItem() {}, removeItem() {} }
    mod.require = (name) => load(name.slice(2))
    mod._compile('const window=module.testWindow,localStorage={setItem(){}},sessionStorage=module.testSessionStorage;\n' +
      ts.transpileModule(fs.readFileSync(file, 'utf8'), {
        compilerOptions: { module: ts.ModuleKind.CommonJS },
      }).outputText, file)
    modules[name] = mod.exports
    return mod.exports
  }
  return { window, auth: load('auth'), runs: load('runs'), chat: load('chat') }
}
async function main() {
  const a = page(), b = page()
  let sid = 'synthetic-session', revoked = false, busy = true, conflict = false
  let stopped = false, timer
  const profile = () => ({ user_id: 'synthetic', session_id: sid, csrf_token: 'csrf' })
  const json = (body, status = 200) => new Response(JSON.stringify(body), { status })
  global.fetch = async (url, options = {}) => {
    if (url === '/api/auth/me') return json(profile(), revoked ? 401 : 200)
    if (url === '/api/auth/login') return json({ ...profile(), tab_token: 'test-tab' })
    if (url === '/api/auth/logout') { revoked = true; return json({}) }
    if (url === '/api/chat/run') return json({ running: busy })
    if (conflict) return json({ detail: '当前账号正在处理另一个问题' }, 409)
    return new Response(new ReadableStream({
      start(controller) {
        timer = setInterval(() => controller.enqueue(new TextEncoder().encode(
          'data: {"type":"token","data":{"content":"测试"}}\n\n')), 10)
        options.signal.addEventListener('abort', () => {
          stopped = true; clearInterval(timer)
          controller.error(new DOMException('Aborted', 'AbortError'))
        }, { once: true })
      }, cancel() { clearInterval(timer) },
    }), { headers: { 'Content-Type': 'text/event-stream' } })
  }
  await a.auth.getCurrentUser(); await b.auth.getCurrentUser()
  let receivedRun
  a.window.addEventListener(a.runs.RUN_STATUS_EVENT, (event) => { receivedRun = event.detail })
  b.runs.broadcastRunStatus()
  await new Promise((resolve) => setTimeout(resolve, 30))
  assert.equal(receivedRun, true)
  busy = false; b.runs.broadcastRunStatus()
  await new Promise((resolve) => setTimeout(resolve, 30))
  assert.equal(receivedRun, false)
  conflict = true
  let expired = 0
  a.window.addEventListener(a.auth.AUTH_EXPIRED_EVENT, () => expired++)
  await assert.rejects(a.chat.chatStream({ query: 'test', thread_id: 'test', user_id: '' }, () => {}), /正在处理另一个问题/)
  assert.equal(expired, 0)
  conflict = false
  let count = 0
  const pending = a.chat.chatStream({ query: 'test', thread_id: 'test', user_id: '' }, () => count++)
    .catch((error) => error)
  while (!count) await new Promise((resolve) => setTimeout(resolve, 10))
  await b.auth.logout()
  assert.match((await pending).message, /登录已失效/)
  assert.ok(stopped); assert.ok(expired)
  await a.auth.logout()
  revoked = false
  await assert.rejects(a.auth.getCurrentUser(), /登录已失效/)
  await a.auth.login('synthetic', 'test')
  sid = 'new-device-session'
  await assert.rejects(a.auth.getCurrentUser(), /登录已失效/)
  console.log('PASS: BroadcastChannel同步任务和退出、取消正在接收的SSE、409不会误判登录失效、拒绝悄悄切换sid')
}
main().catch((error) => { console.error(error); process.exitCode = 1 })
