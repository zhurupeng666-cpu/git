global.window = new EventTarget();
global.localStorage = { setItem() {} };
const fs = require('node:fs');
const Module = require('node:module');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const ts = require(path.join(root, 'node_modules/typescript'));

function loadAuth() {
  const values = new Map([['semiconductor_tab_credentials', JSON.stringify({ session_id: 'test-session', tab_token: 'test-tab' })]]);
  global.sessionStorage = { getItem:key=>values.get(key)||null, setItem:(key,value)=>values.set(key,value), removeItem:key=>values.delete(key) };
  const filename = path.join(root, 'src/api/auth.cjs');
  const mod = new Module(filename);
  mod.filename = filename;
  mod._compile(ts.transpileModule(fs.readFileSync(path.join(root, 'src/api/auth.ts'), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS },
  }).outputText, filename);
  return mod.exports;
}

const profile = { user_id: 'test', username: 'test', display_name: '测试用户', role: 'employee', csrf_token: 'test-csrf', session_id: 'test-session', idle_timeout_seconds: 1800 };
function response(status, body = profile) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

async function run() {
  for (const status of [200, 401]) {
    const auth = loadAuth();
    const calls = [];
    global.fetch = async (url, options) => {
      calls.push({ url, options });
      return response(url.endsWith('/logout') ? status : 200);
    };
    await auth.getCurrentUser(); // 页面刷新时恢复身份和CSRF。
    await auth.logout();
    assert.equal(calls[1].url, '/api/auth/logout');
    assert.equal(calls[1].options.method, 'POST');
    assert.equal(calls[1].options.credentials, 'same-origin');
    assert.equal(new Headers(calls[1].options.headers).get('X-CSRF-Token'), 'test-csrf');
    await assert.rejects(auth.getCsrfToken(), /登录已失效/);
    assert.equal(calls.length, 2); // 退出后缺少Tab Token，不能凭共享Cookie恢复登录。
  }
  for (const status of [403, 500]) {
    const auth = loadAuth();
    let count = 0;
    global.fetch = async (url) => { count++; return response(url.endsWith('/logout') ? status : 200); };
    await auth.getCurrentUser();
    await assert.rejects(auth.logout(), /退出登录失败/);
    assert.equal(await auth.getCsrfToken(), 'test-csrf');
    assert.equal(count, 2); // 退出失败时保留身份，不伪装退出成功。
  }
  console.log('PASS: restored session CSRF, logout POST/cookies, success/expired-session cleanup, failed logout preserves state');
}
run().catch(error => { console.error(error); process.exitCode = 1; });
