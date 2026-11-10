# coding: utf-8
"""
识别结果持久化存储模块 (SQLite)

为 CapsWriter 服务端提供识别结果的历史持久化与检索能力。所有完成识别的
文本、时间戳与任务元数据都会写入本地 SQLite 数据库，供后续查询、统计与
回溯使用。

设计要点：
1. 并发安全：SQLite 连接以 ``check_same_thread=False`` 打开，所有读写操作
   由内部锁串行化，可安全地被 WebSocket 发送协程与 HTTP 查询线程并发访问。
2. WAL 模式：启用 Write-Ahead Logging，写入与读取互不阻塞，显著提升并发
   场景下的吞吐。
3. 全文检索：优先使用 SQLite FTS5 虚拟表做倒排索引检索；在不支持 FTS5 的
   环境下自动降级为 LIKE 前缀匹配，保证可移植性。
4. 参数化查询：所有 SQL 均使用占位符绑定参数，杜绝 SQL 注入。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id       TEXT    NOT NULL,
    source        TEXT    NOT NULL,
    text          TEXT    NOT NULL DEFAULT '',
    text_accu     TEXT    NOT NULL DEFAULT '',
    duration      REAL    NOT NULL DEFAULT 0,
    tokens        TEXT    NOT NULL DEFAULT '[]',
    timestamps    TEXT    NOT NULL DEFAULT '[]',
    time_start    REAL    NOT NULL DEFAULT 0,
    time_submit   REAL    NOT NULL DEFAULT 0,
    time_complete REAL    NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_task_id   ON records(task_id);
CREATE INDEX IF NOT EXISTS idx_records_source     ON records(source);
CREATE INDEX IF NOT EXISTS idx_records_created_at ON records(created_at);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts
    USING fts5(text, text_accu, content='records', content_rowid='id',
               tokenize='trigram');

CREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON records BEGIN
    INSERT INTO records_fts(rowid, text, text_accu)
    VALUES (new.id, new.text, new.text_accu);
END;

CREATE TRIGGER IF NOT EXISTS records_ad AFTER DELETE ON records BEGIN
    INSERT INTO records_fts(records_fts, rowid, text, text_accu)
    VALUES ('delete', old.id, old.text, old.text_accu);
END;

CREATE TRIGGER IF NOT EXISTS records_au AFTER UPDATE ON records BEGIN
    INSERT INTO records_fts(records_fts, rowid, text, text_accu)
    VALUES ('delete', old.id, old.text, old.text_accu);
    INSERT INTO records_fts(rowid, text, text_accu)
    VALUES (new.id, new.text, new.text_accu);
END;
"""


class RecognitionStore:
    """基于 SQLite 的识别结果存储与检索。

    Args:
        db_path: 数据库文件路径；``None`` 时使用内存数据库（便于测试）。

    Example::

        store = RecognitionStore("records.db")
        store.insert(task_id="abc", source="mic", text="你好世界", duration=1.5)
        rows = store.query(keyword="世界")
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._fts_available = False

        if db_path is None:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(db_path, check_same_thread=False)

        self._conn.row_factory = sqlite3.Row
        self._initialize()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)

            # FTS5 为可选增强能力，失败则降级为 LIKE 检索
            try:
                self._conn.executescript(_FTS_SCHEMA)
                self._fts_available = True
            except sqlite3.OperationalError:
                self._fts_available = False
            self._conn.commit()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def insert(self, task_id: str, source: str, text: str = "",
               text_accu: str = "", duration: float = 0.0,
               tokens: Optional[List[str]] = None,
               timestamps: Optional[List[float]] = None,
               time_start: float = 0.0, time_submit: float = 0.0,
               time_complete: float = 0.0) -> int:
        """插入一条识别记录，返回自增主键 id。"""
        tokens_json = json.dumps(tokens or [], ensure_ascii=False)
        timestamps_json = json.dumps(timestamps or [], ensure_ascii=False)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO records
                    (task_id, source, text, text_accu, duration,
                     tokens, timestamps, time_start, time_submit,
                     time_complete, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (task_id, source, text, text_accu, duration,
                 tokens_json, timestamps_json, time_start, time_submit,
                 time_complete, created_at),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def query(self, keyword: Optional[str] = None, source: Optional[str] = None,
              start: Optional[str] = None, end: Optional[str] = None,
              limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """按关键词 / 来源 / 时间范围检索记录，按时间倒序返回。

        Args:
            keyword: 全文检索关键词，匹配 ``text`` 或 ``text_accu``。
            source: 按来源过滤（``mic`` / ``file``）。
            start: 起始时间（ISO 字符串，含）。
            end: 截止时间（ISO 字符串，含）。
        """
        where: List[str] = []
        params: List[Any] = []

        if keyword:
            if self._fts_available and self._is_safe_fts_query(keyword):
                where.append(
                    "id IN (SELECT rowid FROM records_fts WHERE records_fts MATCH ?)"
                )
                params.append(self._build_fts_query(keyword))
            else:
                like = f"%{keyword}%"
                where.append("(text LIKE ? OR text_accu LIKE ?)")
                params.extend([like, like])
        if source:
            where.append("source = ?")
            params.append(source)
        if start:
            where.append("created_at >= ?")
            params.append(start)
        if end:
            where.append("created_at <= ?")
            params.append(end)

        sql = "SELECT * FROM records"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        return [self._row_to_dict(r) for r in rows]

    def count(self, keyword: Optional[str] = None,
              source: Optional[str] = None) -> int:
        """统计符合条件的记录总数。"""
        where: List[str] = []
        params: List[Any] = []

        if keyword:
            if self._fts_available and self._is_safe_fts_query(keyword):
                where.append(
                    "id IN (SELECT rowid FROM records_fts WHERE records_fts MATCH ?)"
                )
                params.append(self._build_fts_query(keyword))
            else:
                like = f"%{keyword}%"
                where.append("(text LIKE ? OR text_accu LIKE ?)")
                params.extend([like, like])
        if source:
            where.append("source = ?")
            params.append(source)

        sql = "SELECT COUNT(*) AS c FROM records"
        if where:
            sql += " WHERE " + " AND ".join(where)

        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return int(row["c"])

    def stats(self) -> Dict[str, Any]:
        """返回存储概况：总记录数、来源分布、时间范围。"""
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) AS c FROM records"
            ).fetchone()["c"]
            by_source = {
                r["source"]: r["c"]
                for r in self._conn.execute(
                    "SELECT source, COUNT(*) AS c FROM records GROUP BY source"
                ).fetchall()
            }
            span = self._conn.execute(
                "SELECT MIN(created_at) AS lo, MAX(created_at) AS hi FROM records"
            ).fetchone()
        return {
            "total": total,
            "by_source": by_source,
            "earliest": span["lo"],
            "latest": span["hi"],
            "fts_available": self._fts_available,
        }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _is_safe_fts_query(keyword: str) -> bool:
        """判断关键词是否适合走 FTS5 trigram 检索。

        trigram 分词器要求查询词长度至少 3 个字符，且仅支持字母、数字、
        中文与空白；不满足时降级为 LIKE 以保证短关键词的正确性。
        """
        compact = "".join(keyword.split())
        if len(compact) < 3:
            return False
        return all(ch.isalnum() or ch.isspace() or "\u4e00" <= ch <= "\u9fff"
                   for ch in keyword)

    @staticmethod
    def _build_fts_query(keyword: str) -> str:
        """将多个词组合为 FTS5 的 AND 查询。"""
        terms = [t for t in keyword.split() if t]
        return " AND ".join(f'"{t}"' for t in terms) or '""'

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """将数据库行转换为可 JSON 序列化的字典。"""
        d = dict(row)
        try:
            d["tokens"] = json.loads(d.get("tokens") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["tokens"] = []
        try:
            d["timestamps"] = json.loads(d.get("timestamps") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["timestamps"] = []
        return d

    def close(self) -> None:
        """关闭数据库连接。"""
        with self._lock:
            self._conn.close()
