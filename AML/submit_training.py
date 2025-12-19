#!/usr/bin/env python3
"""
Simple script to submit training jobs to AML using YAML config files.
This is a wrapper around submit_job.py to make it easier to use.
"""
import yaml
import sys
from pathlib import Path

# Add AML directory to path
sys.path.insert(0, str(Path(__file__).parent))

from submit_job import (
    submit_aml_job,
    AmlConfig,
    NamedInputInfo,
    NamedOutputInfo,
    AssetInfo,
)
import argparse


def load_yaml_and_submit(job_config_path: str):
    """Load job configuration from YAML and submit to AML."""
    
    # Load job config
    with open(job_config_path) as f:
        config = yaml.safe_load(f)
    
    # Load workspace config
    ws_config_path = Path(job_config_path).parent / config["aml_config"]
    with open(ws_config_path) as f:
        ws_config = yaml.safe_load(f)
    
    aml_config = AmlConfig(
        subscription_id=ws_config["subscription_id"],
        resource_group_name=ws_config["resource_group_name"],
        workspace_name=ws_config["workspace_name"],
    )
    
    # Parse input data
    input_data = None
    if config.get("input_data"):
        input_data = []
        for inp in config["input_data"]:
            data_asset = None
            if inp.get("data_asset"):
                data_asset = AssetInfo(**inp["data_asset"])
            input_data.append(
                NamedInputInfo(
                    name=inp["name"],
                    data_asset=data_asset,
                    blob_folder=None,
                )
            )
    
    # Parse output data
    output_data = None
    if config.get("output_data"):
        output_data = []
        for out in config["output_data"]:
            output_data.append(
                NamedOutputInfo(
                    name=out["name"],
                    data_asset=None,
                    blob_folder=None,
                )
            )
    
    print(f"Submitting job: {config['display_name']}")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Command: {config['cmd_line']}")
    print(f"Compute: {config['compute_cluster']}")
    if config.get("tags"):
        print(f"Tags: {config['tags']}")
    if config.get("priority"):
        print(f"Priority: {config['priority']}")
    print("-" * 60)
    
    submit_aml_job(
        experiment_name=config["experiment_name"],
        display_name=config["display_name"],
        aml_config=aml_config,
        cmd_line=config["cmd_line"],
        input_data=input_data,
        output_data=output_data,
        environment_name=config.get("environment_name", "pl_terrain_env"),
        compute_cluster=config.get("compute_cluster", "gpu-cluster"),
        shared_memory=config.get("shared_memory", "12g"),
        docker_container_base=config.get(
            "docker_container_base",
            "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04",
        ),
        aml_env_path=config.get("aml_env_path", "AML/pl_terrain_env.yml"),
        instance_count=config.get("instance_count", 1),
        tags=config.get("tags"),
        priority=config.get("priority"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Submit PL terrain training jobs to AML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit TransUNet-RT training
  python AML/submit_training.py AML/yaml_configs/train_transunet_rt.yaml
  
  # Submit U-Net training
  python AML/submit_training.py AML/yaml_configs/train_unet.yaml
  
  # Submit Optuna hyperparameter search
  python AML/submit_training.py AML/yaml_configs/train_optuna.yaml
        """,
    )
    parser.add_argument(
        "job_config",
        type=str,
        help="Path to job configuration YAML file",
    )
    
    args = parser.parse_args()
    
    if not Path(args.job_config).exists():
        print(f"Error: Job config file not found: {args.job_config}")
        sys.exit(1)
    
    load_yaml_and_submit(args.job_config)


if __name__ == "__main__":
    main()
