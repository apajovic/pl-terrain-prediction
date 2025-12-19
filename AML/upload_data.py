"""
Upload data to AML datastore and register as dataset.
Run this once before submitting training jobs.
"""
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from aml_utils import get_credential
import argparse
from pathlib import Path
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description="Upload data to AML datastore")
    parser.add_argument(
        "--config",
        type=str,
        default="AML/yaml_configs/pl_experiment_workspace.yaml",
        help="Path to workspace config YAML",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Local data directory to upload",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="pl-data",
        help="Name for the registered dataset",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help="Version for the dataset (default: auto-increment)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Load workspace config
    with open(args.config, "r") as f:
        ws_config = yaml.safe_load(f)

    subscription_id = ws_config["subscription_id"]
    resource_group = ws_config["resource_group_name"]
    workspace = ws_config["workspace_name"]

    print(f"Connecting to workspace: {workspace}")
    print(f"Resource group: {resource_group}")
    print(f"Subscription: {subscription_id}")

    ml_client = MLClient(
        get_credential(), subscription_id, resource_group, workspace
    )

    # Create data asset
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")

    print(f"Uploading data from: {data_path}")
    
    my_data = Data(
        path=str(data_path),
        type=AssetTypes.URI_FOLDER,
        description="PL terrain prediction training data",
        name=args.dataset_name,
        version=args.dataset_version,
    )

    # Register the data asset
    registered_data = ml_client.data.create_or_update(my_data)
    print(f"Dataset '{args.dataset_name}' version '{registered_data.version}' registered successfully!")
    print(f"Data URI: {registered_data.path}")
