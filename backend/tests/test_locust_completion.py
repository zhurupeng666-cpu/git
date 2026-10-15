"""用合成HTTP服务验证20用户无总时限运行及完成后自动退出，不访问业务服务。"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, {'user_id': self.server.identities.get(self.headers.get('X-Session-ID')),
                         'role': 'employee'})

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', '0')))
        if self.path == '/api/auth/login':
            user_id, session_id = str(uuid4()), str(uuid4())
            self.server.identities[session_id] = user_id
            self.reply(200, {'user_id': user_id, 'csrf_token': 'synthetic',
                             'tab_token': 'synthetic', 'session_id': session_id})
        elif self.path == '/api/sessions':
            self.reply(201, {'id': str(uuid4())})
        else:
            events = [{'type': 'status', 'data': {'stage': 'rerank'}},
                      {'type': 'source', 'data': []},
                      {'type': 'token', 'data': {'content': '合成回答'}},
                      {'type': 'done', 'data': None}]
            body = ''.join('event: message\ndata: ' + json.dumps(event, ensure_ascii=False) + '\n\n' for event in events).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)


class LocustCompletionTests(unittest.TestCase):
    def test_twenty_samples_stop_without_runtime_cutoff(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.identities = {}
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                metrics = Path(temporary) / 'metrics.csv'
                env = {**os.environ, 'LOCUST_PASSWORD': 'synthetic-only',
                       'LOCUST_ACCOUNT_COUNT': '20', 'LOCUST_ACCOUNT_START': '1',
                       'LOCUST_MODE': 'all', 'LOCUST_RAG_PER_USER': '1',
                       'LOCUST_SSE_DEADLINE': '0', 'LOCUST_STOP_AFTER_SAMPLES': '1',
                       'LOCUST_METRICS_FILE': str(metrics)}
                process = subprocess.Popen([sys.executable, '-m', 'locust', '-f', 'locustfile.py',
                    '--host', f'http://127.0.0.1:{server.server_port}', '--headless', '-u', '20', '-r', '20'],
                    cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                try:
                    output, _ = process.communicate(timeout=60)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                    raise
                self.assertEqual(process.returncode, 0, output.decode('utf-8', errors='replace')[-3000:])
                with metrics.open(encoding='utf-8-sig', newline='') as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 20)
                self.assertEqual(len({row['username'] for row in rows}), 20)
                self.assertTrue(all(row['success'] == 'True' and row['done'] == 'True' for row in rows))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
