# coding: utf-8
"""
服务端 HTTP 监控与检索接口模块

为 CapsWriter 服务端提供一个轻量级的 HTTP 接口层，用于暴露运行状态、
Prometheus 指标以及识别结果检索能力。实现完全基于 Python 标准库
（http.server / urllib / json），零第三方依赖。

提供的接口：
- ``GET /health``        健康检查（JSON）
- ``GET /metrics``       Prometheus 文本格式指标
- ``GET /records``       识别结果检索（JSON，支持 keyword / source / 分页）
- ``GET /records/stats`` 存储概况统计（JSON）

该服务运行在独立的守护线程中，不阻塞主进程的 WebSocket 事件循环。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .metrics import MetricsRegistry
from .storage import RecognitionStore


class _Handler(BaseHTTPRequestHandler):
    """HTTP 请求处理器。

    通过 ``self.server.app`` 访问 MetricsHttpServer 实例以获取依赖。
    """

    server_version = "CapsWriterMetrics/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 (http.server 命名约定)
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        if route == "/health":
            self._handle_health()
        elif route == "/metrics":
            self._handle_metrics()
        elif route == "/records":
            self._handle_records(query)
        elif route == "/records/stats":
            self._handle_stats()
        else:
            self._send_json({"error": "not found"}, status=404)

    def _handle_health(self) -> None:
        app = self.server.app
        payload = {
            "status": "ok",
            "version": app.version,
            "uptime_seconds": round(app.metrics.uptime_seconds(), 3),
            "connections_active": app.metrics.snapshot()["gauges"].get(
                "caps_server_connections_active", 0
            ),
        }
        self._send_json(payload)

    def _handle_metrics(self) -> None:
        app = self.server.app
        body = app.metrics.render_prometheus()
        self._send_text(body, content_type="text/plain; version=0.0.4")

    def _handle_records(self, query: dict) -> None:
        app = self.server.app
        try:
            limit = min(max(int(query.get("limit", "100")), 1), 500)
            offset = max(int(query.get("offset", "0")), 0)
        except ValueError:
            self._send_json({"error": "invalid limit/offset"}, status=400)
            return

        keyword = query.get("keyword") or None
        source = query.get("source") or None
        items = app.store.query(keyword=keyword, source=source,
                                limit=limit, offset=offset)
        total = app.store.count(keyword=keyword, source=source)
        self._send_json({
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": items,
        })

    def _handle_stats(self) -> None:
        app = self.server.app
        self._send_json(app.store.stats())

    # ------------------------------------------------------------------
    # 响应辅助
    # ------------------------------------------------------------------
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, status, "application/json; charset=utf-8")

    def _send_text(self, text: str, content_type: str = "text/plain",
                   status: int = 200) -> None:
        self._send_bytes(text.encode("utf-8"), status,
                         f"{content_type}; charset=utf-8")

    def _send_bytes(self, body: bytes, status: int, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # 由 http.server 默认输出到 stderr，这里静默，避免污染主控制台。
        pass


class MetricsHttpServer:
    """线程化的 HTTP 监控服务。

    Args:
        metrics: 指标注册表实例。
        store: 识别结果存储实例。
        version: 服务端版本号（用于 /health 展示）。
        host: 监听地址，默认仅本机回环地址。
        port: 监听端口。
    """

    def __init__(self, metrics: MetricsRegistry, store: RecognitionStore,
                 version: str = "unknown", host: str = "127.0.0.1",
                 port: int = 6017) -> None:
        self.metrics = metrics
        self.store = store
        self.version = version
        self.host = host
        self.port = port

        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """在守护线程中启动 HTTP 服务。"""
        if self._thread and self._thread.is_alive():
            return

        self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._httpd.app = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="metrics-http-server",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """停止 HTTP 服务。"""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    @property
    def url(self) -> str:
        """返回服务根地址，便于日志打印。"""
        return f"http://{self.host}:{self.port}"
