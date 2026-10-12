"""File-backed model registry: active pointer, approved-version stack, per-version status and audit history.

Statuses: candidate -> approved | rejected ; approved -> rolled_back
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from app.common.jsonutil import dumps, utcnow


class RegistryError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, models_dir: Path):
        self.dir = Path(models_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "registry.json"
        self._lock = threading.RLock()

    # ---- persistence
    def _read(self) -> dict:
        if not self.path.exists():
            return {"active": None, "approved_stack": [], "versions": {}}
        return json.loads(self.path.read_text())

    def _write(self, d: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(dumps(d, indent=1))
        os.replace(tmp, self.path)

    @staticmethod
    def _event(rec: dict, status: str, reason: str) -> None:
        rec["status"] = status
        rec.setdefault("history", []).append({"ts": utcnow(), "status": status, "reason": reason})

    # ---- queries
    def snapshot(self) -> dict:
        with self._lock:
            return self._read()

    def active(self) -> str | None:
        return self.snapshot()["active"]

    def artifact_path(self, version: str) -> Path:
        return self.dir / version

    # ---- mutations
    def register_candidate(self, version: str, metrics: dict, reason: str = "") -> None:
        with self._lock:
            d = self._read()
            rec = d["versions"].setdefault(version, {"registered_at": utcnow(), "metrics": metrics})
            self._event(rec, "candidate", reason or "registered")
            self._write(d)

    def reject(self, version: str, reason: str) -> None:
        with self._lock:
            d = self._read()
            if version not in d["versions"]:
                raise RegistryError(f"unknown version {version}")
            if d["active"] == version:
                raise RegistryError("cannot reject the active version")
            self._event(d["versions"][version], "rejected", reason)
            self._write(d)

    def promote(self, version: str, reason: str) -> None:
        with self._lock:
            d = self._read()
            if version not in d["versions"]:
                raise RegistryError(f"unknown version {version}")
            if not self.artifact_path(version).joinpath("manifest.json").exists():
                raise RegistryError(f"artifact for {version} not found on disk")
            self._event(d["versions"][version], "approved", reason)
            d["approved_stack"] = [v for v in d["approved_stack"] if v != version] + [version]
            d["active"] = version
            self._write(d)

    def rollback(self, reason: str, to: str | None = None) -> str:
        """Re-activate a previously approved version. Returns the new active version."""
        with self._lock:
            d = self._read()
            stack, cur = d["approved_stack"], d["active"]
            if cur is None or len(stack) < 2:
                raise RegistryError("no previously approved version to roll back to")
            if to is None:
                target = stack[-2]
            elif to in stack and to != cur:
                target = to
            else:
                raise RegistryError(f"{to} is not a previously approved version")
            if not self.artifact_path(target).joinpath("manifest.json").exists():
                raise RegistryError(f"artifact for {target} is missing; rollback aborted, active model unchanged")
            idx = stack.index(target)
            for v in stack[idx + 1:]:
                self._event(d["versions"][v], "rolled_back", reason)
            d["approved_stack"] = stack[: idx + 1]
            d["active"] = target
            self._write(d)
            return target
