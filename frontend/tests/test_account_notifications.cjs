// 多页面各自使用不同账户，验证通知不会跨账号或误伤同账号的新会话。
const fs = require('node:fs')
const path = require('node:path')
const Module = require('node:module')
const assert = require('node:assert/strict')
const root = path.resolve(__dirname, '..')
const ts = require(path.join(root, 'node_modules/typescript'))
const channels = [], pages = []
class TestChannel {
  constructor(name) { this.name = name; channels.push(this) }
  postMessage(data) {
    for (const other of channels)
      if (other !== this && other.name === this.name)
        queueMicrotask(() => other.onmessage?.({ data }))
  }
}
const profiles = {
  a: { user_id: 'account-a', session_id: 'session-a', csrf_token: 'csrf-a', tab_token: 'tab-a', idle_timeout_seconds: 1800 },
  b: { user_id: 'account-b', session_id: 'session-b', csrf_token: 'csrf-b', tab_token: 'tab-b', idle_timeout_seconds: 1800 },
}
function page(channelEnabled = true, storedSid = '') {
  const win = new EventTarget(), modules = {}, values = new Map()
  if (channelEnabled) win.BroadcastChannel = TestChannel
  if (storedSid) values.set('semiconductor_tab_credentials', JSON.stringify({ session_id: storedSid, tab_token: storedSid.includes('b') ? 'tab-b' : 'tab-a' }))
  let meCalls = 0, runCalls = 0, selected = 'a', expired = 0, finishDelayed
  const storage = {
    getItem: key => values.get(key) || null,
    setItem: (key, value) => values.set(key, value),
    removeItem: key => values.delete(key),
  }
  const local = { setItem(key, newValue) {
    for (const target of pages) if (target.win !== win) {
      const event = new Event('storage'); event.key = key; event.newValue = newValue
      target.win.dispatchEvent(event)
    }
  } }
  const json = body => new Response(JSON.stringify(body), { status: 200 })
  async function fetch(url, options = {}) {
    if (url === '/api/test-delayed') return new Promise(resolve => { finishDelayed = resolve })
    if (url === '/api/auth/login') {
      selected = JSON.parse(options.body).username
      return json(profiles[selected])
    }
    if (url === '/api/auth/me') {
      meCalls++
      const sid = new Headers(options.headers).get('X-Session-ID')
      const user = Object.values(profiles).find(user => user.session_id === sid) || profiles[selected]
      return json(user)
    }
    if (url === '/api/chat/run') { runCalls++; return json({ running: false }) }
    if (url === '/api/auth/logout') return json({})
    throw new Error(url)
  }
  function load(name) {
    if (modules[name]) return modules[name]
    const file = path.join(root, 'src/api', name + '.ts'), mod = new Module(file)
    mod.win = win; mod.local = local; mod.storage = storage; mod.fetch = fetch
    mod.require = name => load(name.slice(2))
    mod._compile('const window=module.win,localStorage=module.local,sessionStorage=module.storage,fetch=module.fetch;\n' +
      ts.transpileModule(fs.readFileSync(file, 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText, file)
    modules[name] = mod.exports
    return mod.exports
  }
  const auth = load('auth'), runs = load('runs')
  win.addEventListener(auth.AUTH_EXPIRED_EVENT, () => expired++)
  const result = { win, auth, runs, values, expired: () => expired, runCalls: () => runCalls,
    finishDelayed: () => finishDelayed(new Response('{}', { status: 401 })) }
  pages.push(result)
  return result
}
const tick = () => new Promise(resolve => setTimeout(resolve, 20))
async function main() {
  const a = page(), b = page(), aOther = page(true, 'session-a'), fallbackB = page(false)
  await a.auth.login('a', 'test'); await aOther.auth.getCurrentUser()
  await b.auth.login('b', 'test'); await fallbackB.auth.login('b', 'test'); await tick()
  // B的重新登录会影响旧B，但绝不能退出任何A。
  assert.equal(a.expired(), 0); assert.equal(aOther.expired(), 0)
  const bBefore = fallbackB.expired(), bRuns = fallbackB.runCalls()
  a.runs.broadcastRunStatus(); await tick()
  assert.equal(fallbackB.runCalls(), bRuns)
  await a.auth.logout(); await tick()
  assert.ok(aOther.expired()); assert.equal(fallbackB.expired(), bBefore)
  assert.equal(fallbackB.auth.getAuthScope().user_id, 'account-b')
  // 刷新后的页面从标签页sid恢复原账号，即使默认Cookie属于别的账号。
  const restored = page(true, 'session-b')
  assert.equal((await restored.auth.getCurrentUser()).user_id, 'account-b')
  profiles.a = { ...profiles.a, session_id: 'new-session-a' }
  const oldA = page(true, 'session-a')
  // 使用明确的旧profile绑定页面，然后以同账号新sid登录模拟新设备。
  profiles.a.session_id = 'session-a'; await oldA.auth.getCurrentUser()
  profiles.a.session_id = 'new-session-a'
  const newA = page(); await newA.auth.login('a', 'test'); await tick()
  assert.ok(oldA.expired()); assert.equal(restored.expired(), 0)
  // 迟到的旧会话退出通知不得退出同账号的新会话。
  const sender = new TestChannel('semiconductor-auth')
  sender.postMessage({ type: 'logout', user_id: 'account-a', session_id: 'session-a' })
  await tick(); assert.equal(newA.expired(), 0)
  const switching = page()
  await switching.auth.login('a', 'test')
  const delayed = switching.auth.authenticatedFetch('/api/test-delayed')
  await tick()
  await switching.auth.login('b', 'test')
  switching.finishDelayed(); await delayed
  assert.equal(switching.expired(), 0)
  assert.equal(switching.auth.getAuthScope().user_id, 'account-b')
  console.log('PASS: 不同账号登录/退出和任务通知隔离、storage兼容、同账号重新登录、迟到通知及刷新恢复')
}
main().catch(error => { console.error(error); process.exitCode = 1 })
