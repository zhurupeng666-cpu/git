const fs=require('node:fs'),Module=require('node:module'),assert=require('node:assert/strict');
const root=require('node:path').resolve(__dirname,'..').replaceAll('\\','/');
const ts=require(root+'/node_modules/typescript');
const winA=new EventTarget(),winB=new EventTarget();
function storage(origin){return{setItem(key,newValue){for(const target of [winA,winB])if(target!==origin){const event=new Event('storage');event.key=key;event.newValue=newValue;target.dispatchEvent(event)}}}}
function load(name,win,auth){const file=root+'/src/api/'+name+'.cjs';const mod=new Module(file);mod.filename=file;
 mod.require=name=>name==='./auth'?auth:name==='./runs'?{broadcastRunStatus(){}}:null;
 mod._compile('const window=module.testWindow,localStorage=module.testStorage,sessionStorage=module.testSessionStorage;\n'+ts.transpileModule(fs.readFileSync(root+'/src/api/'+name+'.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS}}).outputText,file);
 return mod.exports;
}
// 每个模块拥有独立标签页的window和storage作用域。
function moduleFor(name,win,auth){const file=root+'/src/api/'+name+'.cjs',mod=new Module(file);mod.filename=file;mod.testWindow=win;mod.testStorage=storage(win);mod.testSessionStorage={getItem(){return JSON.stringify({session_id:'synthetic-session',tab_token:'test-tab'})},setItem(){},removeItem(){}};mod.require=name=>name==='./auth'?auth:name==='./runs'?{broadcastRunStatus(){}}:null;
 mod._compile('const window=module.testWindow,localStorage=module.testStorage,sessionStorage=module.testSessionStorage;\n'+ts.transpileModule(fs.readFileSync(root+'/src/api/'+name+'.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS}}).outputText,file);return mod.exports;}
const authA=moduleFor('auth',winA),authB=moduleFor('auth',winB),chatA=moduleFor('chat',winA,authA);
let revoked=false,aborted=false,timer,streamController;
global.fetch=async(url,options={})=>{
 if(url==='/api/auth/me')return new Response(JSON.stringify(revoked?{detail:'请先登录'}:{user_id:'synthetic',session_id:'synthetic-session',csrf_token:'test-csrf'}),{status:revoked?401:200});
 if(url==='/api/auth/logout'){revoked=true;return new Response('{}',{status:200});}
 const encoder=new TextEncoder();
 return new Response(new ReadableStream({start(controller){streamController=controller;timer=setInterval(()=>controller.enqueue(encoder.encode('data: {"type":"token","data":{"content":"合成回复"}}\n\n')),10);
 options.signal.addEventListener('abort',()=>{aborted=true;clearInterval(timer);controller.error(new DOMException('Aborted','AbortError'))},{once:true});},cancel(){clearInterval(timer)}}),{headers:{'Content-Type':'text/event-stream'}});
};
async function run(){
 await authA.getCurrentUser();await authB.getCurrentUser();let chunks=0,expired=0;
 winA.addEventListener(authA.AUTH_EXPIRED_EVENT,()=>expired++);
 const pending=chatA.chatStream({query:'合成测试',thread_id:'test',user_id:''},()=>chunks++).then(()=>null,error=>error);
 while(chunks<2)await new Promise(resolve=>setTimeout(resolve,10));
 await authB.logout();const error=await pending;
 assert.match(error.message,/登录已失效/);assert.ok(aborted);assert.ok(expired);
 const stopped=chunks;await new Promise(resolve=>setTimeout(resolve,40));assert.equal(chunks,stopped);
 // A的CSRF缓存已经被清空，/me直接返回401，此时再次退出仍必须成功。
 await authA.logout();
 console.log('PASS: tab B logout stops tab A in-flight SSE, emits login-page event, and expired tab A can logout without getting stuck');
}
run().catch(error=>{clearInterval(timer);console.error(error);process.exitCode=1});
