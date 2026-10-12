"""Validate the dataset and print its schema/imbalance/split summary."""
import json

import _common  # noqa: F401

from app.common.config import Settings
from app.data.schema import chronological_split, load_dataset

s = Settings.from_env()
df, info = load_dataset(s.data_path, s.sample_frac, s.seed)
print(json.dumps(info, indent=2))
print(json.dumps(chronological_split(df).info, indent=2))
