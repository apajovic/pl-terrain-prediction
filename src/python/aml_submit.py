#!/usr/bin/env python3
"""
submit_to_aml.py
Script to submit training jobs to Azure Machine Learning (AML).
This script:
1. Connects to the AML workspace
2. Creates an environment from requirements.txt
3. Uploads data from the data directory to AML datastore
4. Submits the training job with the specified configuration
"""

import argparse
import os
from pathlib import Path
from azureml.core import (
    Workspace,
    Experiment,
    ScriptRunConfig,
    Environment,
    Dataset,
    Datastore,
)
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import RunConfiguration


def get_or_create_workspace(
    subscription_id=None, resource_group=None, workspace_name=None, config_path=None
):
    """
    Get or create an AML workspace.
    If config_path is provided, loads from config file.
    Otherwise, uses the provided credentials.
    """
    if config_path and os.path.exists(config_path):
        print(f"Loading workspace from config: {config_path}")
        ws = Workspace.from_config(path=config_path)
    elif subscription_id and resource_group and workspace_name:
        print(
            f"Connecting to workspace: {workspace_name} in resource group: {resource_group}"
        )
        ws = Workspace(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
        )
    else:
        print("Loading workspace from default config.json")
        ws = Workspace.from_config()

    print(f"Workspace: {ws.name}, Location: {ws.location}")
    return ws


def create_environment(
    ws, env_name="pl-terrain-env", requirements_path="requirements.txt"
):
    """
    Create or update an AML environment from requirements.txt
    """
    print(f"Creating environment: {env_name}")

    # Create environment from base Docker image
    env = Environment(name=env_name)

    # Use a curated environment as base (includes CUDA support)
    env.docker.enabled = True
    env.docker.base_image = (
        "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04"
    )

    # Add pip dependencies from requirements.txt
    if os.path.exists(requirements_path):
        print(f"Adding dependencies from: {requirements_path}")
        with open(requirements_path, "r") as f:
            requirements = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

        # Add tqdm if not present (used in train.py)
        if "tqdm" not in " ".join(requirements):
            requirements.append("tqdm")
        if "mlflow" not in " ".join(requirements):
            requirements.append("mlflow")
            requirements.append("azureml-mlflow")

        from azureml.core.conda_dependencies import CondaDependencies

        conda_dep = CondaDependencies()
        for req in requirements:
            conda_dep.add_pip_package(req)
        env.python.conda_dependencies = conda_dep
    else:
        print(f"Warning: {requirements_path} not found. Using minimal environment.")

    # Register the environment
    env.register(workspace=ws)
    print(f"Environment '{env_name}' registered successfully")

    return env


def upload_data(ws, local_data_path, datastore_name=None, target_path="data"):
    """
    Upload data directory to AML datastore and register as dataset
    """
    print(f"Uploading data from: {local_data_path}")

    # Get default datastore if not specified
    if datastore_name:
        datastore = Datastore.get(ws, datastore_name)
    else:
        datastore = ws.get_default_datastore()

    print(f"Using datastore: {datastore.name}")

    # Upload the data directory
    if os.path.exists(local_data_path):
        datastore.upload(
            src_dir=local_data_path,
            target_path=target_path,
            overwrite=True,
            show_progress=True,
        )
        print(f"Data uploaded to: {datastore.name}/{target_path}")

        # Create a file dataset from the uploaded data
        dataset_name = "pl-terrain-data"
        try:
            dataset = Dataset.File.from_files(path=(datastore, target_path))
            dataset = dataset.register(
                workspace=ws,
                name=dataset_name,
                description="PL terrain prediction training data",
                create_new_version=True,
            )
            print(f"Dataset registered as: {dataset_name}")
            return dataset
        except Exception as e:
            print(f"Warning: Could not register dataset: {e}")
            return None
    else:
        print(f"Warning: Data path {local_data_path} does not exist. Skipping upload.")
        return None


def get_or_create_compute_target(
    ws, compute_name="gpu-cluster", vm_size="STANDARD_NC6", max_nodes=4, min_nodes=0
):
    """
    Get or create a compute target (GPU cluster)
    """
    try:
        compute_target = ComputeTarget(workspace=ws, name=compute_name)
        print(f"Found existing compute target: {compute_name}")
    except ComputeTargetException:
        print(f"Creating new compute target: {compute_name}")
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size,
            max_nodes=max_nodes,
            min_nodes=min_nodes,
            idle_seconds_before_scaledown=300,
        )
        compute_target = ComputeTarget.create(ws, compute_name, compute_config)
        compute_target.wait_for_completion(show_output=True)

    return compute_target


def submit_training_job(
    ws,
    env,
    compute_target,
    source_directory,
    script_name="train.py",
    config_path=None,
    experiment_name="pl-terrain-training",
    dataset=None,
    args=None,
):
    """
    Submit the training job to AML
    """
    print(f"Submitting training job to experiment: {experiment_name}")

    # Prepare script arguments
    script_args = []
    if config_path:
        script_args.extend(["--config", config_path])
    if args:
        script_args.extend(args)

    # Create run configuration
    src = ScriptRunConfig(
        source_directory=source_directory,
        script=script_name,
        arguments=script_args,
        compute_target=compute_target,
        environment=env,
    )

    # Mount or download the dataset if available
    if dataset:
        # Mount the dataset to a path
        named_input = dataset.as_named_input("training_data").as_mount("/data")
        src.run_config.data_references = {named_input.name: named_input.data_reference}
        print("Dataset will be mounted at: /data")

    # Create experiment and submit
    experiment = Experiment(workspace=ws, name=experiment_name)
    run = experiment.submit(src)

    print(f"Run submitted successfully!")
    print(f"Run ID: {run.id}")
    print(f"Portal URL: {run.get_portal_url()}")

    return run


def main():
    parser = argparse.ArgumentParser(
        description="Submit PL terrain prediction training to Azure ML"
    )

    # Workspace arguments
    parser.add_argument(
        "--workspace-config",
        type=str,
        default="config.json",
        help="Path to AML workspace config.json file",
    )
    parser.add_argument("--subscription-id", type=str, help="Azure subscription ID")
    parser.add_argument("--resource-group", type=str, help="Azure resource group")
    parser.add_argument("--workspace-name", type=str, help="AML workspace name")

    # Environment arguments
    parser.add_argument(
        "--env-name",
        type=str,
        default="pl-terrain-env",
        help="Name for the AML environment",
    )
    parser.add_argument(
        "--requirements",
        type=str,
        default="requirements.txt",
        help="Path to requirements.txt",
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Local data directory to upload",
    )
    parser.add_argument(
        "--datastore-name",
        type=str,
        help="Name of AML datastore (uses default if not specified)",
    )
    parser.add_argument(
        "--skip-data-upload",
        action="store_true",
        help="Skip uploading data (use if data already exists in AML)",
    )

    # Compute arguments
    parser.add_argument(
        "--compute-name",
        type=str,
        default="gpu-cluster",
        help="Name of compute target",
    )
    parser.add_argument(
        "--vm-size",
        type=str,
        default="STANDARD_NC6",
        help="Azure VM size for compute",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=4, help="Maximum nodes in cluster"
    )
    parser.add_argument(
        "--min-nodes", type=int, default=0, help="Minimum nodes in cluster"
    )

    # Training arguments
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="pl-terrain-training",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="src/python",
        help="Source directory containing training script",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="train.py",
        help="Training script name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../../default_config.json",
        help="Training configuration file (relative to source_dir)",
    )
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="Enable Optuna hyperparameter search",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train from scratch (ignore checkpoint)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the run to complete",
    )

    args = parser.parse_args()

    # Get workspace
    ws = get_or_create_workspace(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
        config_path=(
            args.workspace_config if os.path.exists(args.workspace_config) else None
        ),
    )

    # Create environment
    env = create_environment(ws, args.env_name, args.requirements)

    # Upload data
    dataset = None
    if not args.skip_data_upload:
        dataset = upload_data(ws, args.data_dir, args.datastore_name)
    else:
        print("Skipping data upload as requested")
        # Try to get existing dataset
        try:
            dataset = Dataset.get_by_name(ws, "pl-terrain-data")
            print("Using existing dataset: pl-terrain-data")
        except:
            print("Warning: Could not find existing dataset")

    # Get or create compute target
    compute_target = get_or_create_compute_target(
        ws, args.compute_name, args.vm_size, args.max_nodes, args.min_nodes
    )

    # Prepare additional script arguments
    script_args = []
    if args.optuna:
        script_args.append("--optuna")
    if args.from_scratch:
        script_args.append("--from_scratch")

    # Submit training job
    run = submit_training_job(
        ws,
        env,
        compute_target,
        args.source_dir,
        args.script,
        args.config,
        args.experiment_name,
        dataset,
        script_args,
    )

    if args.wait:
        print("Waiting for run to complete...")
        run.wait_for_completion(show_output=True)
        print(f"Run completed with status: {run.get_status()}")
    else:
        print("Run submitted. Use --wait flag to wait for completion.")


if __name__ == "__main__":
    main()
