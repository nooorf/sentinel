# Azure ML deployment (managed online endpoint)

Deployment scripts for serving a trained Sentinel artifact on an Azure ML managed online endpoint, with blue/green slots, a smoke test and traffic-switch rollback.

## Prerequisites
Azure subscription with quota for the chosen CPU SKU · Azure CLI (`az login`) · `pip install -r deployment/azure_ml/requirements.txt` · a trained local artifact (`var/models/<version>`).

```bash
az login && az account set --subscription <SUB_ID>
az group create -n rg-sentinel -l westeurope
az ml workspace create -n mlw-sentinel -g rg-sentinel          # needs: az extension add -n ml
cp deployment/azure_ml/config.example.yaml azure_config.yaml    # edit values (git-ignored)
```

## Deploy (blue slot) and smoke-test
```bash
python deployment/azure_ml/deploy.py deploy --config azure_config.yaml --artifact var/models/<version> --slot blue
python deployment/azure_ml/smoke_test.py   --config azure_config.yaml --artifact var/models/<version>
```
`deploy.py` registers the artifact folder as a model, stages `score.py` + the `app/` package (the pickled preprocessor needs it), builds the environment from `conda.yaml`,
creates the endpoint (key auth) and deployment, and routes traffic only after the deployment succeeds. Get the key: `az ml online-endpoint get-credentials -n <endpoint> -g rg-sentinel -w mlw-sentinel`.

## Update / rollback
Deploy a new version into another slot (`--slot green`) → smoke test → traffic switches to it. **Rollback** = route traffic back:
`python deployment/azure_ml/deploy.py rollback --config azure_config.yaml --slot blue`.

## Cleanup (avoid charges — managed endpoints bill while they exist)
```bash
python deployment/azure_ml/deploy.py delete --config azure_config.yaml
az group delete -n rg-sentinel --yes --no-wait     # removes everything in the group
```
No Azure credentials are needed for local development, tests, or CI.
