global.window = new EventTarget();
global.localStorage = { setItem() {} };
const fs = require('node:fs');
const Module = require('node:module');
const assert = require('node:assert/strict');
const root = require('node:path').resolve(__dirname, '..').replaceAll('\\', '/');
const ts = require(root + '/node_modules/typescript');
function load(name,auth) {
  if(name==='auth') {
    const values=new Map([['semiconductor_tab_credentials',JSON.stringify({session_id:'same-login',tab_token:'test-tab'})]]);
    global.sessionStorage={getItem:key=>values.get(key)||null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
  }
  const filename=root+'/src/api/'+name+'.cjs';
  const mod=new Module(filename);mod.filename=filename;
  mod.require=name=>{if(name==='./auth')return auth;if(name==='./runs')return {broadcastRunStatus(){}};throw new Error(name)};
  mod._compile(ts.transpileModule(fs.readFileSync(root+'/src/api/'+name+'.ts','utf8'),{
    compilerOptions:{module:ts.ModuleKind.CommonJS}
  }).outputText,filename);
  return mod.exports;
}
const profile=token=>({user_id:'test',role:'employee',session_id:'same-login',csrf_token:token});
const json=(status,body)=>new Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json'}});
async function run(){
  // 同一登录会话的CSRF缓存失步时允许恢复，新登录会话不能偷偷替换旧页面身份。
  const auth=load('auth');const chat=load('chat',auth);
  const calls=[];let me=0,posts=0;
  global.fetch=async(url,options)=>{
    calls.push({url,options});
    if(url==='/api/auth/me')return json(200,profile(++me===1?'old-token':'new-token'));
    const token=new Headers(options.headers).get('X-CSRF-Token');
    assert.equal(options.credentials,'same-origin');
    posts++;
    if(posts===1){assert.equal(token,'old-token');return json(403,{detail:'CSRF验证失败'})}
    assert.equal(token,'new-token');
    return new Response('data: {"type":"token","data":{"content":"你好"}}\n\ndata: {"type":"done","data":null}\n\n',{
      headers:{'Content-Type':'text/event-stream'}
    });
  };
  await auth.getCurrentUser();const events=[];
  await chat.chatStream({query:'你好',thread_id:'test',user_id:''},event=>events.push(event));
  assert.deepEqual(events.map(event=>event.type),['token','done']);
  assert.equal(me,2);assert.equal(posts,2);
  // 权限不足和不可信来源不得被当作CSRF错误自动重试。
  for(const detail of ['仅工程师有权执行此操作','请求来源不可信']){
    const api=load('auth');let writes=0,reads=0;
    global.fetch=async(url)=>{if(url==='/api/auth/me'){reads++;return json(200,profile('token'))}writes++;return json(403,{detail})};
    const response=await api.authenticatedFetch('/api/test',{method:'POST',body:'{}'});
    assert.equal(await api.getApiError(response,'未知错误'),detail);
    assert.equal(writes,1);assert.equal(reads,1);
  }
  // 持续失败最多重试一次，防止无限刷新或重复发送。
  const repeated=load('auth');let writes=0;
  global.fetch=async(url)=>url==='/api/auth/me'?json(200,profile('token')):(writes++,json(403,{detail:'CSRF验证失败'}));
  const response=await repeated.authenticatedFetch('/api/test',{method:'POST',body:'{}'});
  assert.equal(response.status,403);assert.equal(writes,2);
  console.log('PASS: rotated CSRF recovers SSE, sends correct Cookie/headers, distinguishes permission/origin rejection, retries once only');
}
run().catch(error=>{console.error(error);process.exitCode=1});
