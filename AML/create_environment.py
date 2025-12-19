from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from aml_utils import get_credential
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource_group", type=str, default="dedicated-rg")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.resource_group == "dedicated-rg":
        workspace = "as-dedicated-w3-ws"
    elif args.resource_group == "playground-rg":
        workspace = "as-playground-w3-ws"
    else:
        raise ValueError(
            "Resource group must be either 'dedicated-rg' or 'playground-rg'."
        )

    subscription_id = "5c9e4789-4852-4ffe-8551-d682affcbd74"

    ml_client = MLClient(
        get_credential(), subscription_id, args.resource_group, workspace
    )
    env_docker_conda = Environment(
        image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:latest",
        conda_file="environment.yml",
        name="prompt_compression",
        description="Environment for speculative decoding. Base image plus conda environment.",
    )
    ml_client.environments.create_or_update(env_docker_conda)
