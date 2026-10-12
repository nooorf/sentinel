from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [to_jsonable(v) for v in o.tolist()]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return None if f != f else f
    if isinstance(o, float):
        return None if o != o else o
    if isinstance(o, Path):
        return str(o)
    return o


def dumps(o, **kw) -> str:
    return json.dumps(to_jsonable(o), **kw)


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
