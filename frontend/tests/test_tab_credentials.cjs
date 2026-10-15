// 独立页面作用域模拟新建、刷新、关闭与活动计时，不使用真实账户或JWT。
const fs = require('node:fs')
const path = require('node:path')
const Module = require('node:module')
const assert = require('node:assert/strict')
const root = path.resolve(__dirname, '..')
const ts = require(path.join(root, 'node_modules/typescript'))
const code = ts.transpileModule(fs.readFileSync(path.join(root, 'src/api/auth.ts'), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS },
}).outputText
const key = 'semiconductor_tab_credentials'

function page(values = new Map()) {
  const window = new EventTarget(), document = new EventTarget(), calls = [], ticks = new Map()
  document.visibilityState = 'visible'
  let now = 100000, timerId = 0, rejected = false
  const user = { user_id: 'synthetic-user', session_id: 'synthetic-session', csrf_token: 'csrf',
    tab_token: 'independent-tab-token', idle_timeout_seconds: 60 }
  const storage = { getItem: key => values.get(key) || null, setItem: (key, value) => values.set(key, value),
    removeItem: key => values.delete(key) }
  const localValues = new Map()
  const local = { setItem: (key, value) => localValues.set(key, value) }
  async function fetch(url, options = {}) {
    calls.push({ url, options })
    if (url === '/api/auth/login') return new Response(JSON.stringify(user))
    const headers = new Headers(options.headers)
    assert.equal(headers.get('X-Session-ID'), user.session_id)
    assert.equal(headers.get('X-Tab-Token'), user.tab_token)
    if (options.method === 'POST') assert.equal(headers.get('X-CSRF-Token'), user.csrf_token)
    if (rejected) return new Response('{}', { status: 401 })
    const { tab_token, ...profile } = user
    return new Response(JSON.stringify(url === '/api/auth/me' ? profile : {}))
  }
  const mod = new Module(path.join(root, 'src/api/auth.test.cjs'))
  mod.filename = path.join(root, 'src/api/auth.test.cjs')
  mod.window = window; mod.document = document; mod.storage = storage; mod.local = local; mod.fetch = fetch
  mod.clock = { now: () => now }
  mod.setInterval = callback => { ticks.set(++timerId, callback); return timerId }
  mod.clearInterval = id => ticks.delete(id)
  mod._compile('const window=module.window,document=module.document,sessionStorage=module.storage,' +
    'localStorage=module.local,fetch=module.fetch,Date=module.clock,setInterval=module.setInterval,clearInterval=module.clearInterval;\n' + code, mod.filename)
  return { auth: mod.exports, window, document, values, calls, localValues,
    advance: value => { now += value }, tick: () => { for (const fn of ticks.values()) fn() },
    reject: () => { rejected = true }, timerCount: () => ticks.size }
}
const flush = () => new Promise(resolve => setImmediate(resolve))
async function main() {
  const empty = page()
  await assert.rejects(empty.auth.getCurrentUser(), /登录已失效/)
  await assert.rejects(empty.auth.authenticatedFetch('/api/test'), /登录已失效/)
  assert.equal(empty.calls.length, 0)
  const legacy = page(new Map([['semiconductor_page_session', 'public-sid-only']]))
  await assert.rejects(legacy.auth.getCurrentUser(), /登录已失效/)
  assert.equal(legacy.calls.length, 0)
  const broken = page(new Map([[key, '{invalid-json']]))
  await assert.rejects(broken.auth.getCurrentUser(), /登录已失效/)
  const loggedIn = page()
  const profile = await loggedIn.auth.login('synthetic', 'test')
  assert.equal(profile.tab_token, undefined)
  const saved = JSON.parse(loggedIn.values.get(key))
  assert.deepEqual(Object.keys(saved).sort(), ['session_id', 'tab_token'])
  assert.equal(saved.tab_token, 'independent-tab-token')
  await loggedIn.auth.authenticatedFetch('/api/test', { method: 'POST' })
  const refreshed = page(new Map(loggedIn.values))
  assert.equal((await refreshed.auth.getCurrentUser()).user_id, 'synthetic-user')
  const newlyOpened = page()
  await assert.rejects(newlyOpened.auth.getCurrentUser(), /登录已失效/)
  assert.equal(newlyOpened.calls.length, 0)
  // 普通关闭后建立空sessionStorage的新页面，不复用原标签页的凭证。
  loggedIn.values.clear()
  const reopened = page(loggedIn.values)
  await assert.rejects(reopened.auth.getCurrentUser(), /登录已失效/)
  console.log('PASS: 新建/关闭后重开不继承登录、同页刷新恢复、旧sid和损坏存储拒绝、请求携带Tab Token')

  const active = page(); await active.auth.login('synthetic', 'test')
  let expired = 0
  active.window.addEventListener(active.auth.AUTH_EXPIRED_EVENT, () => expired++)
  const stop = active.auth.startActivityTracking()
  active.window.dispatchEvent(new Event('pointerdown')); await flush()
  assert.equal(active.calls.filter(call => call.url === '/api/auth/activity').length, 1)
  active.window.dispatchEvent(new Event('keydown')); await flush()
  assert.equal(active.calls.filter(call => call.url === '/api/auth/activity').length, 1)
  active.advance(30000); active.tick(); await flush()
  assert.equal(active.calls.filter(call => call.url === '/api/auth/activity').length, 1)
  assert.ok(active.calls.some(call => call.url === '/api/auth/me'))
  active.document.visibilityState = 'hidden'
  active.advance(31000)
  active.window.dispatchEvent(new Event('keydown')); active.tick(); await flush()
  assert.equal(expired, 1)
  assert.equal(active.values.has(key), false)
  stop(); assert.equal(active.timerCount(), 0)
  const revoked = page(new Map(refreshed.values)); revoked.reject()
  await assert.rejects(revoked.auth.getCurrentUser(), /登录已失效/)
  assert.equal(revoked.values.has(key), false)
  await revoked.auth.logout()
  assert.ok(!JSON.stringify([...revoked.localValues]).includes('independent-tab-token'))
  console.log('PASS: 活动请求节流、后台检查不刷新空闲期、隐藏页超时、401清理与已失效退出')
}
main().catch(error => { console.error(error); process.exitCode = 1 })
