"""Mobile web UI: open http://127.0.0.1:8090 in the phone's browser.

Stdlib only. One conversation per browser tab (session id in the page). Answers stream to the
browser as Server-Sent Events.
"""
import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import llm
from .agent import Agent, Events
from .config import Config

PAGE = (Path(__file__).parent / "static" / "index.html").read_bytes()
CFG = Config()
SESSIONS = {}
LLM_LOCK = threading.Lock()  # llama-server runs a single slot; serialize questions across tabs


def session(sid):
    if sid not in SESSIONS:
        SESSIONS[sid] = Agent(dataclasses.replace(CFG))  # own copy: the think toggle is per tab
    return SESSIONS[sid]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif url.path == "/health":
            self._send(200, "application/json", json.dumps({"llm": llm.health(CFG.llm_url)}).encode())
        elif url.path == "/ask":
            q = parse_qs(url.query)
            self._ask(q.get("sid", [""])[0], q.get("q", [""])[0].strip(), q.get("think", ["0"])[0] == "1")
        elif url.path == "/reset":
            SESSIONS.pop(parse_qs(url.query).get("sid", [""])[0], None)
            self._send(200, "application/json", b"{}")
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _ask(self, sid, question, think):
        if not sid or not question:
            return self._send(400, "text/plain", b"missing sid or q")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(kind, data):
            self.wfile.write(f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        ev = Events(on_status=lambda t: emit("status", t),
                    on_reasoning=lambda d: emit("reasoning", d),
                    on_content=lambda d: emit("content", d),
                    on_answer_start=lambda: emit("answer_start", ""),
                    on_challenge=lambda url: emit("challenge", url))
        agent = session(sid)
        try:
            if not LLM_LOCK.acquire(blocking=False):
                emit("status", "waiting for another question to finish")
                LLM_LOCK.acquire()
            try:
                agent.cfg.think_for_answer = think
                ans = agent.ask(question, ev)
            finally:
                LLM_LOCK.release()
            emit("done", {"text": ans.text, "searches": ans.searches, "usage": ans.usage,
                          "sources": [{"n": s.n, "title": s.title, "url": s.url} for s in ans.sources]})
        except llm.LLMError as e:
            emit("error", str(e))
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser went away


def main():
    host, port = "127.0.0.1", 8090
    print(f"phone-search web UI on http://{host}:{port}  (llm: {CFG.llm_url})")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
