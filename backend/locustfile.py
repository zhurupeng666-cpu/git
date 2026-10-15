import csv
import json
import math
import os
import time
from itertools import count

import gevent
from locust import HttpUser, task, between, events
from locust.clients import HttpSession
from locust.exception import StopUser
from scripts.loadtest_sse import SSEMetrics

# API双Worker不等于Locust分布式Worker；同一Locust进程内按编号分配独立账号。
# 当前方案：20个独立账号，各执行一次真实RAG问答。
ACCOUNT_COUNT = int(os.getenv("LOCUST_ACCOUNT_COUNT", "20"))
ACCOUNT_START = int(os.getenv("LOCUST_ACCOUNT_START", "1"))
RAG_PER_USER = int(os.getenv("LOCUST_RAG_PER_USER", "1"))
MODE = os.getenv("LOCUST_MODE", "all")
AUTH_INTERVAL = float(os.getenv("LOCUST_AUTH_INTERVAL", "1"))
STREAM_DEADLINE = float(os.getenv("LOCUST_SSE_DEADLINE", "330"))
AUTO_STOP = os.getenv("LOCUST_STOP_AFTER_SAMPLES", "0") == "1"
METRICS_FILE = os.getenv("LOCUST_METRICS_FILE", "loadtest_sse_metrics.csv")
ACCOUNT_COUNTER = count(ACCOUNT_START)
RAG_SAMPLES = []
if MODE not in {"all", "auth", "rag"} or min(ACCOUNT_COUNT, AUTH_INTERVAL) <= 0 or STREAM_DEADLINE < 0 or RAG_PER_USER < 0:
    raise ValueError("压测配置无效，请检查LOCUST_MODE、账号数量和超时")

QUESTIONS = [
    "请根据企业知识库说明CVD设备腔体压力异常的原因及排查步骤。",
    "请根据企业知识库分析CMP工艺的常见缺陷和处理方法。",
    "请根据企业知识库说明刻蚀设备常见报警及排查流程。",
    "请根据企业知识库说明FMEA风险评估的主要步骤。"
]


@events.test_start.add_listener
def reset_test(environment, **kwargs):
    """支持重复启动测试，避免上次的账号计数和TTFT混入本次。"""
    global ACCOUNT_COUNTER
    ACCOUNT_COUNTER = count(ACCOUNT_START)
    RAG_SAMPLES.clear()


class SemiconductorUser(HttpUser):
    wait_time = between(0.5, 1.0)

    def on_start(self):
        self.auth_greenlet = None
        self.auth_client = None
        number = next(ACCOUNT_COUNTER)
        if number >= ACCOUNT_START + ACCOUNT_COUNT:
            raise StopUser()
        self.username = f"loadtest{number:03d}"
        self.user_number = number
        self.rag_count = 0
        password = os.getenv("LOCUST_PASSWORD")
        if not password:
            raise RuntimeError("请设置LOCUST_PASSWORD环境变量")
        with self.client.post("/api/auth/login", json={"username": self.username, "password": password},
                              name="/api/auth/login", catch_response=True, timeout=(10, 30)) as response:
            if response.status_code != 200:
                if AUTO_STOP:
                    gevent.spawn_later(0.2, self.environment.runner.quit)
                response.failure(f"登录失败：HTTP {response.status_code}")
                raise StopUser()
            try:
                data = response.json()
                self.user_id = data["user_id"]
                self.headers = {"X-CSRF-Token": data["csrf_token"], "X-Tab-Token": data["tab_token"],
                                "X-Session-ID": data["session_id"]}
            except (KeyError, ValueError):
                response.failure("登录响应缺少JWT会话配套凭证")
                raise StopUser()
            response.success()
        with self.client.post("/api/sessions", json={"title": f"压测会话-{self.username}"}, headers=self.headers,
                              name="/api/sessions", catch_response=True, timeout=(10, 30)) as response:
            if response.status_code != 201:
                if AUTO_STOP:
                    gevent.spawn_later(0.2, self.environment.runner.quit)
                response.failure(f"创建会话失败：HTTP {response.status_code}")
                raise StopUser()
            self.thread_id = response.json()["id"]
            response.success()
        if MODE == "all":
            # 独立HTTP客户端避免长SSE阻塞认证采样；Cookie仅复制当前虚拟用户的登录。
            self.auth_client = HttpSession(self.client.base_url, self.environment.events.request, self)
            self.auth_client.cookies.update(self.client.cookies)
            self.auth_greenlet = gevent.spawn(self.poll_auth)

    def poll_auth(self):
        while True:
            self.check_profile(self.auth_client)
            gevent.sleep(AUTH_INTERVAL)

    def check_profile(self, client):
        with client.get("/api/auth/me", headers=self.headers, name="/api/auth/me",
                        catch_response=True, timeout=(10, 30)) as response:
            if response.status_code != 200:
                response.failure(f"身份校验失败：HTTP {response.status_code}")
            elif response.json().get("user_id") != self.user_id:
                response.failure("身份校验返回了其他账号")
            else:
                response.success()

    def on_stop(self):
        if self.auth_greenlet:
            self.auth_greenlet.kill(block=False)
        if self.auth_client:
            self.auth_client.close()

    @task
    def run_scenario(self):
        if MODE == "auth":
            self.check_profile(self.client)
            return
        if self.rag_count >= RAG_PER_USER:
            return
        self.rag_count += 1
        self.run_rag()

    def run_rag(self):
        start = time.perf_counter()
        metrics = SSEMetrics(started=start)
        error = None
        status = 0
        request_id = ""
        payload = {"query": QUESTIONS[(self.user_number - 1) % len(QUESTIONS)],
                   "thread_id": self.thread_id, "user_id": self.user_id}
        with self.client.post("/api/chat/stream/sse", json=payload, headers=self.headers,
                              name="/api/chat/stream/sse", stream=True, catch_response=True,
                              timeout=(10, None if STREAM_DEADLINE == 0 else 30)) as response:
            status = response.status_code
            request_id = response.headers.get("X-Request-ID", "")
            try:
                if status != 200:
                    error = f"HTTP {status}"
                elif not response.headers.get("Content-Type", "").lower().startswith("text/event-stream"):
                    error = "响应类型不是text/event-stream"
                else:
                    # 0关闭客户端总时长限制；本轮收到done/error后才结束该请求。
                    with gevent.Timeout(STREAM_DEADLINE or None, False):
                        for chunk in response.iter_content(chunk_size=None):
                            if not chunk:
                                continue
                            metrics.feed(chunk)
                            if metrics.completed or metrics.error:
                                break
                    if STREAM_DEADLINE > 0 and (time.perf_counter() - start) >= STREAM_DEADLINE and not metrics.completed:
                        error = "SSE超过总耗时上限"
                    else:
                        error = metrics.failure()
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:240]}"
            finally:
                # 统计的是完整SSE耗时和实际字节数，不能只统计响应头或Content-Length。
                total_ms = (time.perf_counter() - start) * 1000
                response.request_meta["response_time"] = total_ms
                response.request_meta["response_length"] = metrics.response_length
                if error:
                    response.failure(error)
                else:
                    response.success()
                response.close()
                row = {"username": self.username, "http_status": status, "request_id": request_id,
                       "total_ms": round(total_ms, 2), "ttft_ms": round(metrics.first_token_ms, 2) if metrics.first_token_ms is not None else "",
                       "token_events": metrics.token_events, "source_events": metrics.source_events,
                       "done": metrics.completed, "success": not bool(error), "last_stage": metrics.last_stage,
                       "stages_ms": json.dumps(metrics.stages, ensure_ascii=False), "error": error or ""}
                RAG_SAMPLES.append(row)
                print(f"[{self.username}] HTTP={status} 完整耗时={total_ms:.0f}ms "
                      f"TTFT={row['ttft_ms']}ms Token={metrics.token_events} done={metrics.completed} "
                      f"最后阶段={metrics.last_stage} request_id={request_id} 错误={error}", flush=True)

        # 最后一条响应退出统计上下文后停止，避免依赖固定运行时长截断长请求。
        if AUTO_STOP and len(RAG_SAMPLES) >= ACCOUNT_COUNT:
            gevent.spawn_later(0.2, self.environment.runner.quit)


@events.test_stop.add_listener
def print_ttft_summary(environment, **kwargs):
    """首Token指标单独输出，不能作为额外HTTP请求污染QPS和失败率。"""
    if not RAG_SAMPLES:
        print("本次未完成RAG样本，请检查模式、运行时长和账号配置")
        return
    with open(METRICS_FILE, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(RAG_SAMPLES[0]))
        writer.writeheader()
        writer.writerows(RAG_SAMPLES)
    successful = [float(row["ttft_ms"]) for row in RAG_SAMPLES if row["success"] and row["ttft_ms"] != ""]
    values = sorted(successful)
    print(f"RAG样本={len(RAG_SAMPLES)} 成功={sum(row['success'] for row in RAG_SAMPLES)} "
          f"首Token样本={sum(row['ttft_ms'] != '' for row in RAG_SAMPLES)}，明细：{METRICS_FILE}")
    if values:
        p95 = values[max(0, math.ceil(len(values) * 0.95) - 1)]
        print(f"成功RAG TTFT平均={sum(values)/len(values):.0f}ms P95={p95:.0f}ms")
    else:
        print("TTFT无成功样本，请查看明细中的HTTP状态、业务错误和最后阶段")
