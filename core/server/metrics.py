# coding: utf-8
"""
服务端运行时指标采集模块 (Metrics)

为 CapsWriter 服务端提供线程安全的运行时指标采集能力，支持三种指标类型：
- counter   : 单调递增的累计计数（如连接总数、任务总数、错误总数）
- gauge     : 可增可减的瞬时值（如活跃连接数、队列长度、RTF）
- histogram : 数值分布（如识别耗时、实时率 RTF），输出 _sum/_count/_bucket

指标通过 :class:`MetricsRegistry` 集中管理，并可直接渲染为 Prometheus
文本格式（text/plain; version=0.0.4），供外部监控系统（Prometheus / Grafana）
抓取。

设计要点：
1. 线程安全：WebSocket 接收、发送与 HTTP 服务分别运行在不同线程/协程，
   所有读写均通过内部锁保护。
2. 零依赖：仅使用标准库 threading / time / collections。
3. 主进程聚合：识别耗时等指标通过 Result 中的时间戳在主进程侧计算，
   避免在识别子进程与主进程之间引入额外的共享内存结构。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# 识别耗时直方图的预设分桶（单位：秒）
_DURATION_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _format_key(name: str, labels: Optional[Dict[str, str]] = None) -> str:
    """将指标名与标签组合为内部唯一键。

    Prometheus 数据模型中的「指标名 + 标签集合」在这里被折叠为单个字符串键，
    以保证注册表可以用普通 dict 实现，从而获得 O(1) 的读写复杂度。
    """
    if not labels:
        return name
    parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{parts}}}"


class MetricsRegistry:
    """线程安全的指标注册表。

    用法::

        metrics = MetricsRegistry()
        metrics.increment("connections_total")
        metrics.set_gauge("connections_active", 3)
        metrics.observe("recognition_duration_seconds", 0.42)
        print(metrics.render_prometheus())
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._started_monotonic = time.monotonic()

    # ------------------------------------------------------------------
    # counter
    # ------------------------------------------------------------------
    def increment(self, name: str, value: float = 1.0,
                  labels: Optional[Dict[str, str]] = None) -> None:
        """对 counter 累加 ``value``。"""
        key = _format_key(name, labels)
        with self._lock:
            self._counters[key] += value

    # ------------------------------------------------------------------
    # gauge
    # ------------------------------------------------------------------
    def set_gauge(self, name: str, value: float,
                  labels: Optional[Dict[str, str]] = None) -> None:
        """设置 gauge 的瞬时值。"""
        key = _format_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def inc_gauge(self, name: str, value: float = 1.0,
                  labels: Optional[Dict[str, str]] = None) -> None:
        """对 gauge 进行增量调整（可为负）。"""
        key = _format_key(name, labels)
        with self._lock:
            self._gauges[key] = self._gauges.get(key, 0.0) + value

    # ------------------------------------------------------------------
    # histogram
    # ------------------------------------------------------------------
    def observe(self, name: str, value: float,
                labels: Optional[Dict[str, str]] = None) -> None:
        """记录一次观测值（用于 histogram）。"""
        key = _format_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    # ------------------------------------------------------------------
    # 快照与导出
    # ------------------------------------------------------------------
    def uptime_seconds(self) -> float:
        """返回服务端已运行的秒数。"""
        return time.monotonic() - self._started_monotonic

    def snapshot(self) -> Dict[str, dict]:
        """返回所有指标的结构化快照，供 /metrics 或调试使用。"""
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {k: list(v) for k, v in self._histograms.items()}
        return {
            "counters": counters,
            "gauges": gauges,
            "histograms": histograms,
            "uptime_seconds": self.uptime_seconds(),
        }

    def render_prometheus(self) -> str:
        """渲染为 Prometheus 文本格式。

        格式遵循 text/plain; version=0.0.4：每行一个样本，
        histogram 展开为 ``_count`` / ``_sum`` / ``_bucket`` 三个子指标。
        """
        lines: List[str] = []

        # 内置的 uptime 指标
        lines.append("# HELP caps_server_uptime_seconds Server uptime in seconds.")
        lines.append("# TYPE caps_server_uptime_seconds gauge")
        lines.append(f"caps_server_uptime_seconds {self.uptime_seconds():.3f}")

        with self._lock:
            counters = sorted(self._counters.items())
            gauges = sorted(self._gauges.items())
            histograms = sorted(self._histograms.items())

        # counter：需同时给出 # TYPE（按 base name 去重）
        seen_types: set = set()
        for key, value in counters:
            base = key.split("{", 1)[0]
            if base not in seen_types:
                lines.append(f"# TYPE {base} counter")
                seen_types.add(base)
            lines.append(f"{key} {value:g}")

        for key, value in gauges:
            base = key.split("{", 1)[0]
            if base not in seen_types:
                lines.append(f"# TYPE {base} gauge")
                seen_types.add(base)
            lines.append(f"{key} {value:g}")

        for key, values in histograms:
            base = key.split("{", 1)[0]
            if base not in seen_types:
                lines.append(f"# TYPE {base} histogram")
                seen_types.add(base)
            if not values:
                continue
            lines.append(f"{key}_count {len(values)}")
            lines.append(f"{key}_sum {sum(values):g}")

            # 分桶计数（累计）
            buckets = sorted(values)
            idx = 0
            for bound in _DURATION_BUCKETS:
                while idx < len(buckets) and buckets[idx] <= bound:
                    idx += 1
                lines.append(f"{key}_bucket{{le=\"{bound}\"}} {idx}")
            lines.append(f"{key}_bucket{{le=\"+Inf\"}} {len(values)}")

        return "\n".join(lines) + "\n"
