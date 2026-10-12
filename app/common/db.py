"""Small SQLite helper (stdlib). One connection per operation; WAL mode; safe across threads."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, source TEXT NOT NULL, batch_tag TEXT,
  model_version TEXT, features TEXT NOT NULL, prob REAL, anomaly REAL, score REAL NOT NULL,
  risk TEXT, flagged INTEGER NOT NULL, label INTEGER);
CREATE INDEX IF NOT EXISTS ix_events_source ON events(source, id);
CREATE INDEX IF NOT EXISTS ix_events_tag ON events(batch_tag, id);
CREATE TABLE IF NOT EXISTS monitor_reports (
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, model_version TEXT, source TEXT, batch_tag TEXT,
  status TEXT NOT NULL, drift_detected INTEGER, summary TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS retraining_runs (
  id TEXT PRIMARY KEY, trigger TEXT NOT NULL, status TEXT NOT NULL, requested_at TEXT NOT NULL,
  started_at TEXT, finished_at TEXT, reason TEXT, drift_report_id TEXT, overrides TEXT,
  candidate_version TEXT, candidate_metrics TEXT, active_version TEXT, active_metrics TEXT,
  decision TEXT, error TEXT, transitions TEXT NOT NULL);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        try:
            yield c
            c.commit()
        finally:
            c.close()
