"""Deploy / roll back / delete the Sentinel model on an Azure ML managed online endpoint (SDK v2).

STATUS: written against azure-ai-ml SDK v2 but NOT executed against a live subscription by the author.
Requires:  pip install -r deployment/azure_ml/requirements.txt  and  `az login`.

  python deployment/azure_ml/deploy.py deploy   --config azure_config.yaml --artifact var/models/<version> [--slot green]
  python deployment/azure_ml/deploy.py rollback --config azure_config.yaml --slot blue
  python deployment/azure_ml/deploy.py delete   --config azure_config.yaml
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def client(cfg) -> MLClient:
    return MLClient(DefaultAzureCredential(), cfg["subscription_id"], cfg["resource_group"], cfg["workspace"])


def stage_code() -> Path:
    out = HERE / "build" / "code"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copytree(ROOT / "app", out / "app", ignore=shutil.ignore_patterns("__pycache__", "dashboard", "api"))
    shutil.copy(HERE / "score.py", out / "score.py")
    return out


def deploy(cfg, artifact: Path, slot: str):
    ml = client(cfg)
    ep = ManagedOnlineEndpoint(name=cfg["endpoint_name"], auth_mode="key", description="Sentinel fraud detection (portfolio)")
    ml.online_endpoints.begin_create_or_update(ep).result()
    model = ml.models.create_or_update(Model(path=str(artifact), name=cfg["model_name"], type="custom_model",
                                             description=f"Sentinel artifact {artifact.name}",
                                             tags={"sentinel_version": artifact.name}))
    env = Environment(name="sentinel-inference", image=cfg["base_image"], conda_file=str(HERE / "conda.yaml"))
    dep = ManagedOnlineDeployment(name=slot, endpoint_name=cfg["endpoint_name"], model=model, environment=env,
                                  code_configuration=CodeConfiguration(code=str(stage_code()), scoring_script="score.py"),
                                  instance_type=cfg["instance_type"], instance_count=cfg["instance_count"])
    ml.online_deployments.begin_create_or_update(dep).result()
    ep = ml.online_endpoints.get(cfg["endpoint_name"])
    ep.traffic = {slot: 100}  # switch all traffic only after the deployment succeeded
    ml.online_endpoints.begin_create_or_update(ep).result()
    print(f"deployed model v{model.version} to slot '{slot}' and routed 100% traffic")


def rollback(cfg, slot: str):
    ml = client(cfg)
    ep = ml.online_endpoints.get(cfg["endpoint_name"])
    if slot not in [d.name for d in ml.online_deployments.list(cfg["endpoint_name"])]:
        sys.exit(f"deployment slot '{slot}' does not exist")
    ep.traffic = {slot: 100}
    ml.online_endpoints.begin_create_or_update(ep).result()
    print(f"traffic routed to '{slot}'")


def delete(cfg):
    client(cfg).online_endpoints.begin_delete(name=cfg["endpoint_name"]).result()
    print("endpoint (and its deployments) deleted")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["deploy", "rollback", "delete"])
    ap.add_argument("--config", default="azure_config.yaml")
    ap.add_argument("--artifact")
    ap.add_argument("--slot", default="blue")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.action == "deploy":
        if not a.artifact:
            sys.exit("--artifact is required")
        deploy(cfg, Path(a.artifact), a.slot)
    elif a.action == "rollback":
        rollback(cfg, a.slot)
    else:
        delete(cfg)
