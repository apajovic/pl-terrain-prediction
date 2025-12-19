# Quick Start: Running Training on Azure ML

## Example: Train TransUNet-RT model

This example shows how to run:
```bash
python src/python/train.py --config configs/transunet-rt.json
```
on Azure Machine Learning.

### Step 1: Configure Your Azure ML Workspace

Edit `AML/yaml_configs/pl_experiment_workspace.yaml`:

```yaml
subscription_id: YOUR_SUBSCRIPTION_ID          # Find in Azure Portal
resource_group_name: YOUR_RESOURCE_GROUP       # Your resource group name
workspace_name: YOUR_WORKSPACE_NAME            # Your AML workspace name
```

### Step 2: One-Time Setup - Upload Data

Run this **once** to upload your data to AML:

```bash
./AML/run_aml_training.sh --upload-data
```

Or manually:
```bash
python AML/upload_data.py \
    --config AML/yaml_configs/pl_experiment_workspace.yaml \
    --data-dir data \
    --dataset-name pl-terrain-data
```

### Step 3: Submit Training Job

#### Option A: Using the convenience script (RECOMMENDED)

```bash
./AML/run_aml_training.sh AML/yaml_configs/train_transunet_rt.yaml
```

#### Option B: Using Python directly

```bash
python AML/submit_training.py AML/yaml_configs/train_transunet_rt.yaml
```

#### Option C: First time (upload data + train in one command)

```bash
./AML/run_aml_training.sh --upload-data AML/yaml_configs/train_transunet_rt.yaml
```

### What Happens Next?

1. The script will:
   - Create/update the conda environment with all dependencies
   - Submit the training job to your GPU cluster
   - Print the AML Studio URL
   - Open the URL in your browser

2. In AML Studio you can:
   - Monitor training progress in real-time
   - View logs and metrics
   - Download trained models
   - Compare experiments

## Other Training Examples

### Train U-Net model
```bash
./AML/run_aml_training.sh AML/yaml_configs/train_unet.yaml
```

### Train PMNet v3
```bash
./AML/run_aml_training.sh AML/yaml_configs/train_pmnetv3.yaml
```

### Run Optuna hyperparameter search
```bash
./AML/run_aml_training.sh AML/yaml_configs/train_optuna.yaml
```

## Custom Configuration

To create your own training job, copy an existing YAML config:

```bash
cp AML/yaml_configs/train_transunet_rt.yaml AML/yaml_configs/train_my_model.yaml
```

Then edit it:

```yaml
aml_config: pl_experiment_workspace.yaml
experiment_name: my_custom_experiment
display_name: my_custom_training_run
cmd_line: >
  python src/python/train.py \
    --config configs/my-config.json \
    --from_scratch

input_data:
  - name: training_data
    data_asset:
      name: pl-terrain-data
      version: "latest"
    blob_folder: null

output_data:
  - name: model_output
    data_asset: null
    blob_folder: null

environment_name: pl_terrain_env
compute_cluster: gpu-cluster              # Change this to your cluster name
shared_memory: 12g
docker_container_base: mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04
aml_env_path: AML/pl_terrain_env.yml
instance_count: 1                         # Use >1 for distributed training
```

Then submit:
```bash
./AML/run_aml_training.sh AML/yaml_configs/train_my_model.yaml
```

## Troubleshooting

### "Compute cluster not found"
Create a GPU cluster in Azure ML or update the `compute_cluster` field in your YAML config.

### "Dataset not found"
Make sure you've uploaded data first:
```bash
./AML/run_aml_training.sh --upload-data
```

### "Authentication failed"
Login to Azure:
```bash
az login
```

### Check job status
Visit the AML Studio URL printed by the script, or use:
```bash
az ml job list --resource-group YOUR_RG --workspace-name YOUR_WS
```

## Advanced: Using the Standalone Script

If you prefer the original approach:

```bash
# Upload data
python src/python/aml_submit.py \
    --workspace-config AML/yaml_configs/pl_experiment_workspace.yaml \
    --data-dir data

# Submit training
python src/python/aml_submit.py \
    --workspace-config AML/yaml_configs/pl_experiment_workspace.yaml \
    --config configs/transunet-rt.json \
    --skip-data-upload \
    --experiment-name pl-terrain-transunet \
    --wait
```
