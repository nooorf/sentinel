"""Azure ML managed online endpoint scoring script. Reuses the exact Predictor used by the local API
(same preprocessing, same models, same threshold). The `app/` package is shipped alongside this file
because the pickled preprocessor references `app.features.preprocess.log_amount`.

Request:  {"transactions": [ {V1..V28, Amount}, ... ]}   (or a single transaction object)
"""
import json
import os
from pathlib import Path

import pandas as pd

predictor = None


def init():
    global predictor
    from app.models.predictor import Predictor

    root = Path(os.environ["AZUREML_MODEL_DIR"])
    manifest = next(root.rglob("manifest.json"))  # model was registered from one artifact folder
    predictor = Predictor.load(manifest.parent)


def run(raw_data):
    payload = json.loads(raw_data)
    rows = payload["transactions"] if isinstance(payload, dict) and "transactions" in payload else payload
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows or len(rows) > 500:
        raise ValueError("send 1-500 transactions")
    return {"model_version": predictor.version, "predictions": predictor.predict_frame(pd.DataFrame(rows))}
