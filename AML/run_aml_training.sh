#!/bin/bash
# Quick setup and submission script for AML training

set -e

echo "=========================================="
echo "PL Terrain Prediction - AML Job Submission"
echo "=========================================="
echo ""

# Check if workspace config exists
WORKSPACE_CONFIG="AML/yaml_configs/pl_experiment_workspace.yaml"
if [ ! -f "$WORKSPACE_CONFIG" ]; then
    echo "Error: Workspace config not found: $WORKSPACE_CONFIG"
    echo "Please edit $WORKSPACE_CONFIG with your Azure ML workspace details."
    exit 1
fi

# Parse arguments
UPLOAD_DATA=false
JOB_CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --upload-data)
            UPLOAD_DATA=true
            shift
            ;;
        --job)
            JOB_CONFIG="$2"
            shift 2
            ;;
        *)
            JOB_CONFIG="$1"
            shift
            ;;
    esac
done

# Upload data if requested
if [ "$UPLOAD_DATA" = true ]; then
    echo "Step 1: Uploading data to AML..."
    python AML/upload_data.py \
        --config "$WORKSPACE_CONFIG" \
        --data-dir data \
        --dataset-name pl-terrain-data
    echo ""
fi

# Submit job if config provided
if [ -n "$JOB_CONFIG" ]; then
    echo "Step 2: Submitting training job..."
    echo "Job config: $JOB_CONFIG"
    python AML/submit_training.py "$JOB_CONFIG"
else
    echo "Available job configurations:"
    echo "  - AML/yaml_configs/train_transunet_rt.yaml  (TransUNet Real-Time)"
    echo "  - AML/yaml_configs/train_unet.yaml          (U-Net)"
    echo "  - AML/yaml_configs/train_pmnetv3.yaml       (PMNet v3)"
    echo "  - AML/yaml_configs/train_optuna.yaml        (Optuna HPO)"
    echo ""
    echo "Usage:"
    echo "  ./AML/run_aml_training.sh [--upload-data] AML/yaml_configs/train_transunet_rt.yaml"
    echo ""
    echo "Examples:"
    echo "  # First time: upload data and train"
    echo "  ./AML/run_aml_training.sh --upload-data AML/yaml_configs/train_transunet_rt.yaml"
    echo ""
    echo "  # Subsequent runs: just train"
    echo "  ./AML/run_aml_training.sh AML/yaml_configs/train_unet.yaml"
fi
