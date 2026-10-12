"""Repeatable latency/throughput benchmark. Writes reports/benchmark.json + .md. Every number comes from this run.

Model-only latency  : Predictor.predict_frame in-process (preprocessing + all models in the decision path).
HTTP end-to-end     : real requests to a running API (use --spawn to start uvicorn on a free port).
"""
import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _common  # noqa: F401
import httpx
import numpy as np
import pandas as pd

from app.common.config import Settings
from app.data.schema import FEATURES
from app.models.predictor import Predictor
from app.registry.registry import ModelRegistry

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=None)
ap.add_argument("--spawn", action="store_true", help="start uvicorn locally for the HTTP benchmark")
ap.add_argument("--iterations", type=int, default=300)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--batch-size", type=int, default=100)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
s = Settings.from_env()
reg = ModelRegistry(s.models_dir)
if not reg.active():
    raise SystemExit("No active model. Run scripts/train.py first.")
pred = Predictor.load(reg.artifact_path(reg.active()))
pool = pd.read_csv(Path(pred.path) / "pool.csv.gz")  # real held-out feature rows (unlabelled)
rng = np.random.default_rng(a.seed)
rows = pool.sample(n=min(len(pool), 2000), random_state=a.seed)[FEATURES]


def pct(v):
    v = np.asarray(v)
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
            "mean": float(v.mean()), "n": int(len(v))}


def timeit(fn, n, warm):
    for _ in range(warm):
        fn()
    out = []
    for _ in range(n):
        t = time.perf_counter(); fn(); out.append((time.perf_counter() - t) * 1000)
    return out


idx = rng.integers(0, len(rows), a.iterations + a.warmup + 1)
it = iter(range(10**9))
single = lambda: pred.predict_frame(rows.iloc[[idx[next(it) % len(idx)]]])  # noqa: E731
batch_df = rows.iloc[: a.batch_size]
res = {"model_version": pred.version, "decision_model": pred.decision_model,
       "model_only": {"single": pct(timeit(single, a.iterations, a.warmup)),
                      f"batch_{a.batch_size}": pct(timeit(lambda: pred.predict_frame(batch_df), max(20, a.iterations // 10), 5))}}

proc = None
url = a.url
if a.spawn:
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    env = {**os.environ, "SENTINEL_DATA_DIR": str(s.data_dir)}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api.main:app", "--port", str(port), "--log-level", "warning"], env=env)
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            if httpx.get(url + "/health", timeout=1).json().get("model_loaded"):
                break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
if url:
    try:
        payloads = [{k: float(v) for k, v in r.items()} for r in rows.iloc[:200].to_dict("records")]
        bpay = {"transactions": payloads[: a.batch_size]}
        with httpx.Client(timeout=30) as c:
            j = iter(range(10**9))
            one = lambda: c.post(url + "/predict", json=payloads[next(j) % len(payloads)]).raise_for_status()  # noqa: E731
            res["http_single_sequential"] = pct(timeit(one, a.iterations, a.warmup))
            res[f"http_batch_{a.batch_size}"] = pct(timeit(lambda: c.post(url + "/predict/batch", json=bpay).raise_for_status(),
                                                           max(20, a.iterations // 10), 5))
        N = a.iterations * 2
        def worker(i, cl):
            t = time.perf_counter(); cl.post(url + "/predict", json=payloads[i % len(payloads)]).raise_for_status()
            return (time.perf_counter() - t) * 1000
        with httpx.Client(timeout=30) as cl:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(a.concurrency) as ex:
                lat = list(ex.map(lambda i: worker(i, cl), range(N)))
            wall = time.perf_counter() - t0
        res["http_concurrent"] = {**pct(lat), "concurrency": a.concurrency, "requests": N, "throughput_rps": N / wall}
    finally:
        if proc:
            proc.terminate()
res["environment"] = {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
                      "machine": platform.machine(), "single_request_rows": 1, "batch_rows": a.batch_size,
                      "iterations": a.iterations, "warmup": a.warmup, "seed": a.seed, "uvicorn_workers": 1,
                      "note": "client and server share this machine; the HTTP client is a Python thread pool"}
out = Path("reports"); out.mkdir(exist_ok=True)
(out / "benchmark.json").write_text(json.dumps(res, indent=1))
f = lambda d: f"{d['p50']:.2f} / {d['p95']:.2f} / {d['p99']:.2f}"  # noqa: E731
L = ["# Benchmark (generated by scripts/benchmark.py)", "", f"Model `{pred.version}` ({pred.decision_model}). Latency in ms: p50 / p95 / p99.", "",
     "| Scenario | p50 / p95 / p99 (ms) | n |", "|---|---|---|"]
for k, v in res.items():
    if isinstance(v, dict) and "p50" in v:
        L.append(f"| {k} | {f(v)} | {v['n']} |")
for k, v in res["model_only"].items():
    L.append(f"| model_only {k} | {f(v)} | {v['n']} |")
if "http_concurrent" in res:
    h = res["http_concurrent"]; L.append(f"\nThroughput at concurrency {h['concurrency']}: **{h['throughput_rps']:.1f} req/s** ({h['requests']} requests).")
L += ["", "Environment: `" + json.dumps(res["environment"]) + "`"]
(out / "benchmark.md").write_text("\n".join(L)); print("\n".join(L))
