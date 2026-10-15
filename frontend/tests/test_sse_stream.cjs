const fs = require('fs');
const assert = require('node:assert/strict');
const Module = require('module');
const root = require('node:path').resolve(__dirname, '..').replaceAll('\\', '/');
const ts = require(root + '/node_modules/typescript');
const source = fs.readFileSync(root + '/src/api/chat.ts', 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS }});
const mod = new Module(root + '/src/api/chat.cjs');
mod.filename = root + '/src/api/chat.cjs';
mod._compile(compiled.outputText, mod.filename);

async function run() {
  const encoder = new TextEncoder();
  const payload = 'event: start\ndata: {"thread_id":"x"}\n\n'
    + ': heartbeat\n\n'
    + 'event: message\ndata: {"type":"token","data":{"content":"中文\\n"}}\n\n'
    + 'event: message\r\ndata: {"type":"source","data":[]}\r\n\r\n'
    + 'event: message\ndata: {"type":"error","data":{"message":"测试错误"}}\n\n'
    + 'event: message\ndata: {"type":"done","data":null}\n\n';
  const bytes = encoder.encode(payload);
  for (const size of [1, 7, bytes.length]) {
    const events = [];
    global.fetch = async () => new Response(new ReadableStream({
      start(controller) {
        for (let i = 0; i < bytes.length; i += size) controller.enqueue(bytes.slice(i, i + size));
        controller.close();
      }
    }));
    await mod.exports.chatStream({query:'test'}, e => events.push(e));
    assert.deepEqual(events.map(e => e.type), ['token', 'source', 'error', 'done']);
    assert.equal(events[0].data.content, '中文\n');
  }
  // 首个 token 到达时，服务器还没有关闭流。
  let controller;
  global.fetch = async () => new Response(new ReadableStream({start(c) { controller = c; }}));
  const events = [];
  const pending = mod.exports.chatStream({query:'test'}, e => events.push(e));
  await new Promise(resolve => setTimeout(resolve, 0));
  controller.enqueue(encoder.encode('event: message\ndata: {"type":"token","data":{"content":"first"}}\n\n'));
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(events[0].data.content, 'first');
  controller.enqueue(encoder.encode('data: {"type":"done","data":null}\n\n'));
  await pending;
  global.fetch = async () => new Response('', {status:500});
  await assert.rejects(mod.exports.chatStream({}, () => {}), /500/);
  global.fetch = async () => new Response('data: {"type":"token","data":{"content":"partial"}}\n\n');
  await assert.rejects(mod.exports.chatStream({}, () => {}), /中断/);
  console.log('PASS: named SSE events, fragmented UTF-8/JSON, CRLF, heartbeat, live first token, HTTP failure, premature EOF');
}
run().catch(e => { console.error(e); process.exitCode = 1; });
