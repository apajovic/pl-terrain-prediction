# Azure Machine Learning (AML) Submission Guide

This directory contains scripts and configuration files for submitting PL terrain prediction training jobs to Azure Machine Learning.

## Setup

### 1. Configure Your Workspace

Edit `AML/yaml_configs/pl_experiment_workspace.yaml` with your Azure ML workspace details:

```yaml
subscription_id: YOUR_SUBSCRIPTION_ID
resource_group_name: YOUR_RESOURCE_GROUP
workspace_name: YOUR_WORKSPACE_NAME
```

### 2. Install Required Packages

```bash
pip install azure-ai-ml azure-identity azure-core azureml-core azureml-mlflow
```

### 3. Authenticate to Azure

```bash
az login
```

## Usage

### Step 1: Upload Data (One-time setup)

Upload your local data directory to AML datastore:

```bash
python AML/upload_data.py \
    --config AML/yaml_configs/pl_experiment_workspace.yaml \
    --data-dir data \
    --dataset-name pl-terrain-data
```

This creates a dataset named `pl-terrain-data` that can be used by training jobs.

### Step 2: Submit Training Jobs

Use the existing `submit_job.py` script with job configuration YAML files:

#### Example: Train TransUNet-RT

```bash
python AML/submit_job.py \
    --aml_config AML/yaml_configs/pl_experiment_workspace.yaml \
    --experiment_name pl_terrain_prediction \
    --display_name train_transunet_rt \
    --cmd_line "python src/python/train.py --config configs/transunet-rt.json" \
    --input_data '[{"name": "training_data", "data_asset": {"name": "pl-terrain-data", "version": "latest"}, "blob_folder": null}]' \
    --output_data '[{"name": "model_output", "data_asset": null, "blob_folder": null}]' \
    --environment_name pl_terrain_env \
    --compute_cluster gpu-cluster \
    --shared_memory 12g \
    --docker_container_base mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04 \
    --aml_env_path AML/pl_terrain_env.yml \
    --instance_count 1
```

Or use the pre-configured YAML files (simpler approach):

```bash
# Load config from YAML and submit
python -c "
import yaml
from jsonargparse import CLI
from submit_job import submit_aml_job, AmlConfig, NamedInputInfo, NamedOutputInfo, AssetInfo

# Load job config
with open('AML/yaml_configs/train_transunet_rt.yaml') as f:
    config = yaml.safe_load(f)

# Load workspace config
with open(f'AML/yaml_configs/{config[\"aml_config\"]}') as f:
    ws_config = yaml.safe_load(f)

aml_config = AmlConfig(
    subscription_id=ws_config['subscription_id'],
    resource_group_name=ws_config['resource_group_name'],
    workspace_name=ws_config['workspace_name']
)

# Parse input data
input_data = [
    NamedInputInfo(
        name=inp['name'],
        data_asset=AssetInfo(**inp['data_asset']) if inp.get('data_asset') else None,
        blob_folder=None
    ) for inp in config.get('input_data', [])
] if config.get('input_data') else None

# Parse output data
output_data = [
    NamedOutputInfo(
        name=out['name'],
        data_asset=None,
        blob_folder=None
    ) for out in config.get('output_data', [])
] if config.get('output_data') else None

submit_aml_job(
    experiment_name=config['experiment_name'],
    display_name=config['display_name'],
    aml_config=aml_config,
    cmd_line=config['cmd_line'],
    input_data=input_data,
    output_data=output_data,
    environment_name=config['environment_name'],
    compute_cluster=config['compute_cluster'],
    shared_memory=config['shared_memory'],
    docker_container_base=config['docker_container_base'],
    aml_env_path=config['aml_env_path'],
    instance_count=config['instance_count']
)
"
```

#### Available Job Configurations

- `train_transunet_rt.yaml` - Train TransUNet with Real-Time config
- `train_unet.yaml` - Train U-Net model
- `train_pmnetv3.yaml` - Train PMNet v3 model
- `train_optuna.yaml` - Hyperparameter search with Optuna

### Step 3: Monitor Your Job

After submission, the script will:
1. Print the AML Studio URL
2. Automatically open the URL in your browser (if possible)

You can monitor training progress, view logs, and download outputs from the AML Studio.

## Alternative: Use the Standalone Submission Script

You can also use the standalone script `src/python/aml_submit.py`:

```bash
# First upload (one-time)
python src/python/aml_submit.py \
    --workspace-config AML/yaml_configs/pl_experiment_workspace.yaml \
    --data-dir data

# Then submit training job
python src/python/aml_submit.py \
    --workspace-config AML/yaml_configs/pl_experiment_workspace.yaml \
    --config configs/transunet-rt.json \
    --skip-data-upload \
    --wait
```

## Compute Cluster

Make sure you have a compute cluster named `gpu-cluster` in your workspace, or update the `compute_cluster` field in the YAML files to match your cluster name.

To create a GPU cluster via Azure CLI:

```bash
az ml compute create --name gpu-cluster \
    --type amlcompute \
    --size STANDARD_NC6 \
    --min-instances 0 \
    --max-instances 4 \
    --resource-group YOUR_RESOURCE_GROUP \
    --workspace-name YOUR_WORKSPACE_NAME
```

## Environment

The training environment is defined in `AML/pl_terrain_env.yml` and includes:
- PyTorch with CUDA support
- All required dependencies from `requirements.txt`
- MLflow for experiment tracking

## Outputs

Training outputs (models, metrics, plots) will be saved to the workspace's default datastore and can be accessed through AML Studio under the experiment runs.
