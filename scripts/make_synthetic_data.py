"""Write SYNTHETIC credit-card-schema data (for CI / smoke tests ONLY - never for benchmark results)."""
import argparse
from pathlib import Path

import _common  # noqa: F401

from app.data.synthetic import make_synthetic_creditcard

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="var/synthetic_creditcard.csv")
ap.add_argument("--n", type=int, default=6000)
a = ap.parse_args()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
make_synthetic_creditcard(a.n).to_csv(a.out, index=False)
print(f"wrote SYNTHETIC data to {a.out} (not real fraud data; for tests/CI only)")
