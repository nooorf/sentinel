"""Persistence for monitoring events, drift reports and retraining runs (SQLite)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.common.db import Database
from app.common.jsonutil import dumps, utcnow
from app.data.schema import FEATURES

SOURCES = ("live", "external", "simulated")


class Store:
    def __init__(self, db_path: Path):
        self.db = Database(db_path)

    # ---------------- events
    def add_events(self, rows: list[dict], source: str, batch_tag: str | None, model_version: str,
                   labels: list[int | None] | None = None) -> int:
        """rows: dicts with features + prob/anomaly/score/risk/flagged."""
        assert source in SOURCES
        labels = labels or [None] * len(rows)
        ts = utcnow()
        data = [(ts, source, batch_tag, model_version, json.dumps({f: r["features"][f] for f in FEATURES}),
                 r["prob"], r["anomaly"], r["score"], r["risk"], int(r["flagged"]), lab)
                for r, lab in zip(rows, labels, strict=True)]
        with self.db.conn() as c:
            c.executemany("INSERT INTO events(ts,source,batch_tag,model_version,features,prob,anomaly,score,risk,"
                          "flagged,label) VALUES (?,?,?,?,?,?,?,?,?,?,?)", data)
        return len(data)

    def recent_events(self, n: int, source: str | None = None, batch_tag: str | None = None,
                      model_version: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM events WHERE 1=1", []
        for col, val in (("source", source), ("batch_tag", batch_tag), ("model_version", model_version)):
            if val is not None:
                q += f" AND {col}=?"
                args.append(val)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(n)
        with self.db.conn() as c:
            rows = c.execute(q, args).fetchall()
        if not rows:
            return pd.DataFrame(columns=[*FEATURES, "score", "flagged", "label", "source", "batch_tag"])
        df = pd.DataFrame([dict(r) for r in rows]).iloc[::-1].reset_index(drop=True)
        feats = pd.DataFrame([json.loads(s) for s in df["features"]])
        return pd.concat([feats, df.drop(columns=["features"])], axis=1)

    def event_stats(self) -> dict:
        with self.db.conn() as c:
            by_src = {r["source"]: r["n"] for r in c.execute("SELECT source, COUNT(*) n FROM events GROUP BY source")}
            live = c.execute("SELECT COUNT(*) n, COALESCE(SUM(flagged),0) f FROM events WHERE source='live'").fetchone()
            risk = {r["risk"]: r["n"] for r in c.execute(
                "SELECT risk, COUNT(*) n FROM events WHERE source='live' GROUP BY risk")}
            labeled = c.execute("SELECT COUNT(*) n FROM events WHERE label IS NOT NULL").fetchone()["n"]
        return {"events_by_source": by_src, "live_predictions": live["n"], "live_flagged": live["f"],
                "live_risk_distribution": risk, "labeled_events": labeled}

    def labeled_events(self, sources=("live", "external")) -> pd.DataFrame:
        """Real (non-simulated) events with ground truth, in arrival order."""
        q = (f"SELECT features, label FROM events WHERE label IS NOT NULL AND source IN ({','.join('?' * len(sources))})"
             " ORDER BY id")
        with self.db.conn() as c:
            rows = c.execute(q, list(sources)).fetchall()
        if not rows:
            return pd.DataFrame()
        feats = pd.DataFrame([json.loads(r["features"]) for r in rows])
        feats["Class"] = [r["label"] for r in rows]
        return feats

    def count_labeled(self, model_version: str | None, label: int = 1, sources=("live", "external")) -> int:
        """Labelled real events with this label among the traffic scored by `model_version`."""
        q = (f"SELECT COUNT(*) n FROM events WHERE label=? AND model_version IS ? "
             f"AND source IN ({','.join('?' * len(sources))})")
        with self.db.conn() as c:
            return int(c.execute(q, [label, model_version, *sources]).fetchone()["n"])

    # ---------------- drift reports
    def save_report(self, rid: str, summary: dict) -> None:
        with self.db.conn() as c:
            c.execute("INSERT INTO monitor_reports VALUES (?,?,?,?,?,?,?,?)",
                      (rid, utcnow(), summary.get("model_version"), summary.get("source"), summary.get("batch_tag"),
                       summary["status"], None if summary.get("drift_detected") is None else int(summary["drift_detected"]),
                       dumps(summary)))

    def get_report(self, rid: str) -> dict | None:
        with self.db.conn() as c:
            r = c.execute("SELECT summary FROM monitor_reports WHERE id=?", (rid,)).fetchone()
        return json.loads(r["summary"]) if r else None

    def latest_report(self) -> dict | None:
        with self.db.conn() as c:
            r = c.execute("SELECT summary FROM monitor_reports ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
        return json.loads(r["summary"]) if r else None

    def list_reports(self, limit: int = 20) -> list[dict]:
        with self.db.conn() as c:
            rows = c.execute("SELECT summary FROM monitor_reports ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(r["summary"]) for r in rows]

    # ---------------- retraining runs
    _JSON_COLS = ("overrides", "candidate_metrics", "active_metrics", "transitions")

    def create_run(self, run: dict) -> None:
        with self.db.conn() as c:
            c.execute(
                "INSERT INTO retraining_runs(id,trigger,status,requested_at,reason,drift_report_id,overrides,"
                "active_version,transitions) VALUES (?,?,?,?,?,?,?,?,?)",
                (run["id"], run["trigger"], run["status"], run["requested_at"], run.get("reason"),
                 run.get("drift_report_id"), dumps(run.get("overrides") or {}), run.get("active_version"),
                 dumps(run["transitions"])))

    def update_run(self, rid: str, **fields) -> None:
        sets, args = [], []
        for k, v in fields.items():
            sets.append(f"{k}=?")
            args.append(dumps(v) if k in self._JSON_COLS else v)
        args.append(rid)
        with self.db.conn() as c:
            c.execute(f"UPDATE retraining_runs SET {','.join(sets)} WHERE id=?", args)

    def get_run(self, rid: str) -> dict | None:
        with self.db.conn() as c:
            r = c.execute("SELECT * FROM retraining_runs WHERE id=?", (rid,)).fetchone()
        return self._run(r) if r else None

    def list_runs(self, limit: int = 20) -> list[dict]:
        with self.db.conn() as c:
            rows = c.execute("SELECT * FROM retraining_runs ORDER BY requested_at DESC, rowid DESC LIMIT ?",
                             (limit,)).fetchall()
        return [self._run(r) for r in rows]

    def active_runs(self) -> list[dict]:
        with self.db.conn() as c:
            rows = c.execute("SELECT * FROM retraining_runs WHERE status IN "
                             "('pending','checking','training','evaluating')").fetchall()
        return [self._run(r) for r in rows]

    def _run(self, r) -> dict:
        d = dict(r)
        for k in self._JSON_COLS:
            d[k] = json.loads(d[k]) if d[k] else None
        return d
