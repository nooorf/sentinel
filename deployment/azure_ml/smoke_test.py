"""Post-deployment smoke test: endpoint provisioning state + one real prediction (uses a dataset example from the artifact).
STATUS: not executed against a live endpoint by the author.   python smoke_test.py --config azure_config.yaml --artifact var/models/<v>"""
import argparse
import json
import tempfile
from pathlib import Path

import yaml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="azure_config.yaml")
ap.add_argument("--artifact", required=True)
ap.add_argument("--slot", default=None)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
ml = MLClient(DefaultAzureCredential(), cfg["subscription_id"], cfg["resource_group"], cfg["workspace"])
ep = ml.online_endpoints.get(cfg["endpoint_name"])
print("provisioning_state:", ep.provisioning_state, "traffic:", ep.traffic)
assert ep.provisioning_state == "Succeeded", "endpoint not healthy"
ex = json.loads((Path(a.artifact) / "examples.json").read_text())[0]["features"]
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump({"transactions": [ex]}, f)
resp = ml.online_endpoints.invoke(endpoint_name=cfg["endpoint_name"], request_file=f.name, deployment_name=a.slot)
out = json.loads(resp)
print(out)
assert out["predictions"][0]["model_version"] == out["model_version"]
print("SMOKE OK")
