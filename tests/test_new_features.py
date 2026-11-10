# coding: utf-8
"""
新增功能单元测试

覆盖三个新增模块：
- core.server.metrics   : 指标采集与 Prometheus 渲染
- core.server.storage   : SQLite 持久化与检索（含中文 FTS5 / LIKE 降级）
- core.server.http_api  : HTTP 健康检查 / 指标 / 检索接口

运行方式（在项目根目录）：
    python3.11 -m unittest discover -s tests -v
"""

import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

# 将项目根目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.server.metrics import MetricsRegistry  # noqa: E402
from core.server.storage import RecognitionStore  # noqa: E402
from core.server.http_api import MetricsHttpServer  # noqa: E402
from core.server.schema import Result  # noqa: E402
from core.server.connection.ws_send import record_result  # noqa: E402


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.m = MetricsRegistry()

    def test_counter_increment(self):
        self.m.increment("tasks_total")
        self.m.increment("tasks_total", 2)
        self.m.increment("tasks_total", labels={"type": "mic"})
        snap = self.m.snapshot()
        self.assertEqual(snap["counters"]["tasks_total"], 3)
        self.assertEqual(snap["counters"]['tasks_total{type="mic"}'], 1)

    def test_gauge_set_and_inc(self):
        self.m.set_gauge("connections_active", 3)
        self.m.inc_gauge("connections_active", 1)
        self.m.inc_gauge("connections_active", -2)
        self.assertEqual(
            self.m.snapshot()["gauges"]["connections_active"], 2
        )

    def test_histogram_observe(self):
        self.m.observe("recognition_duration_seconds", 0.42)
        self.m.observe("recognition_duration_seconds", 2.0)
        snap = self.m.snapshot()
        hist = snap["histograms"]["recognition_duration_seconds"]
        self.assertEqual(len(hist), 2)

    def test_render_prometheus_format(self):
        self.m.increment("tasks_total", labels={"type": "file"})
        self.m.set_gauge("connections_active", 1)
        self.m.observe("recognition_duration_seconds", 0.42)
        text = self.m.render_prometheus()
        self.assertIn("# TYPE caps_server_uptime_seconds gauge", text)
        self.assertIn("# TYPE tasks_total counter", text)
        self.assertIn('tasks_total{type="file"} 1', text)
        self.assertIn("recognition_duration_seconds_count 1", text)
        self.assertIn('recognition_duration_seconds_bucket{le="0.5"} 1', text)

    def test_thread_safety(self):
        def bump():
            for _ in range(1000):
                self.m.increment("concurrent")

        threads = [threading.Thread(target=bump) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.m.snapshot()["counters"]["concurrent"], 8000)


class TestStorage(unittest.TestCase):
    def setUp(self):
        # 使用内存数据库隔离每个用例
        self.store = RecognitionStore(db_path=None)

    def tearDown(self):
        self.store.close()

    def test_insert_and_query_by_keyword_chinese_fts(self):
        self.store.insert(
            task_id="t1", source="mic",
            text="你好世界，这是离线语音输入测试", duration=1.5,
        )
        # >=3 字符走 trigram FTS
        self.assertEqual(self.store.count(keyword="语音输入"), 1)
        # 2 字符走 LIKE 降级
        self.assertEqual(self.store.count(keyword="世界"), 1)
        self.assertEqual(self.store.count(keyword="离线"), 1)

    def test_query_by_source_and_pagination(self):
        for i in range(3):
            self.store.insert(task_id=f"m{i}", source="mic", text=f"麦克风第{i}句")
        for i in range(2):
            self.store.insert(task_id=f"f{i}", source="file", text=f"文件第{i}句")
        self.assertEqual(self.store.count(source="mic"), 3)
        rows = self.store.query(source="mic", limit=2, offset=1)
        self.assertEqual(len(rows), 2)

    def test_tokens_roundtrip(self):
        self.store.insert(
            task_id="t2", source="mic", text="你好",
            tokens=["你", "好"], timestamps=[0.1, 0.2],
        )
        row = self.store.query(keyword="你好")[0]
        self.assertEqual(row["tokens"], ["你", "好"])
        self.assertEqual(row["timestamps"], [0.1, 0.2])

    def test_stats(self):
        self.store.insert(task_id="a", source="mic", text="一")
        self.store.insert(task_id="b", source="file", text="二")
        stats = self.store.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["by_source"], {"mic": 1, "file": 1})

    def test_fts_available_flag(self):
        # 内存库中 trigram FTS 应可用（取决于编译环境，两种结果都合法）
        self.assertIsInstance(self.store.stats()["fts_available"], bool)


class TestHttpApi(unittest.TestCase):
    PORT = 16017

    @classmethod
    def setUpClass(cls):
        cls.metrics = MetricsRegistry()
        cls.store = RecognitionStore(db_path=None)
        cls.store.insert(task_id="h1", source="mic", text="你好世界语音输入")
        cls.server = MetricsHttpServer(
            cls.metrics, cls.store, version="test-1.0",
            host="127.0.0.1", port=cls.PORT,
        )
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.store.close()

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.PORT}{path}") as resp:
            return resp.status, resp.read().decode("utf-8")

    def test_health(self):
        status, body = self._get("/health")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["version"], "test-1.0")

    def test_metrics_endpoint(self):
        status, body = self._get("/metrics")
        self.assertEqual(status, 200)
        self.assertIn("caps_server_uptime_seconds", body)

    def test_records_endpoint(self):
        status, body = self._get("/records?keyword=" + quote("语音输入"))
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], 1)
        self.assertIn("你好世界语音输入", data["items"][0]["text"])

    def test_records_stats_endpoint(self):
        status, body = self._get("/records/stats")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(data["total"], 1)

    def test_not_found(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(f"http://127.0.0.1:{self.PORT}/nope")
        self.assertEqual(ctx.exception.code, 404)


class TestRecordResultIntegration(unittest.TestCase):
    """验证 ws_send 中「指标埋点 + 持久化」的集成逻辑。"""

    def setUp(self):
        self.metrics = MetricsRegistry()
        self.store = RecognitionStore(db_path=None)
        self.app = SimpleNamespace(metrics=self.metrics, store=self.store)

    def tearDown(self):
        self.store.close()

    def test_final_result_records_metrics_and_persists(self):
        result = Result(
            task_id="task-1", socket_id="sock-1", type="mic",
            duration=2.0, time_start=0.0, time_submit=1.0, time_complete=1.5,
            text="你好世界", text_accu="你好世界",
            tokens=["你", "好", "世", "界"],
            timestamps=[0.0, 0.5, 1.0, 1.5],
            is_final=True,
        )
        record_result(self.app, result)

        snap = self.metrics.snapshot()
        self.assertEqual(
            snap["counters"]['caps_server_tasks_total{type="mic"}'], 1
        )
        self.assertAlmostEqual(
            snap["gauges"]['caps_server_rtf{type="mic"}'], 0.25
        )
        self.assertEqual(
            len(snap["histograms"]["caps_server_recognition_duration_seconds"]),
            1,
        )

        # 最终结果应写入持久化存储
        self.assertEqual(self.store.count(), 1)
        row = self.store.query()[0]
        self.assertEqual(row["text"], "你好世界")
        self.assertEqual(row["tokens"], ["你", "好", "世", "界"])

    def test_non_final_result_records_metrics_but_not_persisted(self):
        result = Result(
            task_id="task-2", socket_id="sock-1", type="mic",
            duration=1.0, time_submit=1.0, time_complete=1.2, is_final=False,
        )
        record_result(self.app, result)

        self.assertEqual(
            self.metrics.snapshot()["counters"]['caps_server_tasks_total{type="mic"}'],
            1,
        )
        # 非最终结果不应持久化
        self.assertEqual(self.store.count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
