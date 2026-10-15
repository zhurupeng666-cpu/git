"""压测专用SSE解码和计时，不把首Token指标计入HTTP请求数量。"""
import codecs
import json
import re
import time


class SSEMetrics:
    def __init__(self, started=None, clock=time.perf_counter):
        self.clock = clock
        self.started = clock() if started is None else started
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buffer = ""
        self.first_token_ms = None
        self.token_events = 0
        self.source_events = 0
        self.completed = False
        self.error = None
        self.response_length = 0
        self.stages = {}
        self.last_stage = "response_headers"

    def feed(self, chunk):
        self.response_length += len(chunk)
        self.buffer += self.decoder.decode(chunk)
        while not self.completed and not self.error:
            boundary = re.search(r"\r?\n\r?\n", self.buffer)
            if not boundary:
                return
            frame = self.buffer[:boundary.start()]
            self.buffer = self.buffer[boundary.end():]
            self.parse(frame)

    def parse(self, frame):
        # event只描述SSE信封；业务事件优先读取JSON.type，不能把message当作token。
        event_name = "message"
        data_lines = []
        for line in frame.splitlines():
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].removeprefix(" "))
        if not data_lines:
            return
        text = "\n".join(data_lines)
        if text == "[DONE]":
            self.completed = True
            return
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("SSE data必须是JSON对象")
        kind = payload.get("type") or payload.get("event") or event_name
        data = payload.get("data", payload)
        if kind == "token":
            if isinstance(data, dict) and isinstance(data.get("content"), str) and data["content"]:
                self.token_events += 1
                if self.first_token_ms is None:
                    self.first_token_ms = (self.clock() - self.started) * 1000
        elif kind == "source":
            self.source_events += 1
        elif kind == "status" and isinstance(data, dict):
            self.last_stage = str(data.get("stage") or "unknown")
            self.stages.setdefault(self.last_stage, (self.clock() - self.started) * 1000)
        elif kind == "done":
            self.completed = True
        elif kind == "error":
            self.error = str(data.get("message") or data.get("detail") or "SSE内部错误") if isinstance(data, dict) else str(data)

    def failure(self):
        if self.error:
            return "SSE错误：" + self.error
        if not self.completed:
            return "SSE连接结束但未收到done事件"
        if self.first_token_ms is None:
            return "SSE收到done但没有有效token"
        return None
