import asyncio
from getpass import getpass
import httpx

BASE_URL = "http://127.0.0.1:8000"

async def main():
    """使用同一登录会话测试RAG并发锁是否生效。"""
    username = input("请输入测试用户名：")
    password = getpass("请输入测试密码：")
    async with httpx.AsyncClient(base_url=BASE_URL, trust_env=False, timeout=180) as client:
        # 只登录一次，两个请求共享同一份认证Cookie
        # 查看登录失败的具体原因，不输出密码或JWT
        login = await client.post("/api/auth/login", json={"username": username, "password": password})
        print("登录状态：", login.status_code)
        print("登录响应：", login.text)
        login.raise_for_status()
        user = login.json()
        headers = {"X-CSRF-Token": user["csrf_token"]}
        # 创建一个真实的数据库会话
        response = await client.post("/api/sessions", json={"title": "RAG并发锁测试"}, headers=headers)
        response.raise_for_status()
        session_id = response.json()["id"]
        # 两个请求使用同一用户身份及同一会话
        request_a = {"query": "请详细分析CVD设备压力异常的原因、排查步骤及维护策略，提供完整工程分析。", "thread_id": session_id, "user_id": user["user_id"]}
        request_b = {"query": "CMP工艺有哪些常见缺陷？", "thread_id": session_id, "user_id": user["user_id"]}
        # 保持请求A的SSE连接打开
        async with client.stream("POST", "/api/chat/stream/sse", json=request_a, headers=headers) as first:
            print("请求A状态：", first.status_code)
            if first.status_code != 200:
                print("请求A启动失败")
                return
            # 等待A开始返回SSE事件
            async for line in first.aiter_lines():
                if line.startswith("data:"):
                    break
            # 在请求A尚未完成时发送请求B
            async with client.stream("POST", "/api/chat/stream/sse", json=request_b, headers=headers) as second:
                print("请求B状态：", second.status_code)
                print("并发锁HTTP测试：", "通过" if second.status_code == 409 else "需要进一步检查")

if __name__ == "__main__":
    asyncio.run(main())