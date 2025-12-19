from azure.ai.ml import MLClient, Output, command, Input
from azure.ai.ml.constants import AssetTypes, InputOutputModes
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from jsonargparse import CLI
import aml_utils


@dataclass
class AmlConfig:
    """Configuration for AML resources."""

    subscription_id: str  # Azure subscription ID
    resource_group_name: str  # Azure resource group name
    workspace_name: str  # AML workspace name


@dataclass
class BlobContainerInfo:
    """Info about a user defined blob container."""

    subscription_id: str
    resource_group: str
    storage_account: str
    name: str


@dataclass
class AssetInfo:
    """Info about a workspace model asset."""

    name: str
    version: str

@dataclass
class BlobFolderInfo:
    """Info about a folder on the blob storage."""

    container: BlobContainerInfo
    path: str

@dataclass
class NamedInputInfo:
    """Info about an asset used as named input."""

    name: str
    # Provide BlobFolderInfo for input from blob storage.
    # Provide ModelAssetInfo for input from workspace's default datastore.
    data_asset: Optional[AssetInfo]
    blob_folder: Optional[BlobFolderInfo]


@dataclass
class NamedOutputInfo:
    """Info about an asset used as named output."""

    name: str

    data_asset: Optional[AssetInfo]
    # Provide BlobFolderInfo for output to blob storage.
    # Omit it for output to the root of the workspace's default datastore.
    blob_folder: Optional[BlobFolderInfo]


def get_blob_storage_uri(blob_folder_info: BlobFolderInfo):
    """Returns AML URI for a blob storage location."""
    blob_container_info = blob_folder_info.container
    blob_path = blob_folder_info.path
    assert Path(blob_path) != Path(""), "Blob path must not be empty."
    return (
        f"wasbs://{blob_container_info.name}@{blob_container_info.storage_account}.blob.core.windows.net"
        f"/{blob_path}"
    )

def resolve_input(
    input_data: NamedInputInfo,
    ml_client: MLClient,
) -> Input:
    """Returns a named input object for given named input information."""
    if input_data.data_asset is not None:
        # Input from workspace's defined assets
        assert input_data.blob_folder is None, "Cannot provide both data asset and blob folder."
        
        input_data_uri = ml_client.data.get(input_data.data_asset.name, version=input_data.data_asset.version).path
        return Input(type=AssetTypes.URI_FOLDER, path=input_data_uri, mode=InputOutputModes.RO_MOUNT)
    elif input_data.blob_folder is not None:
        # Input from user defined blob storage.
        input_data_uri = get_blob_storage_uri(input_data.blob_folder)
        return Input(path=input_data_uri)
    else:
        raise "Must provide either model asset or blob folder for input."


def resolve_output(
    output_data: NamedOutputInfo,
    ml_client: MLClient,
) -> Output:
    """Returns a named output object for given named output information."""
    if output_data.data_asset is not None:
        # Output to data asset defined in workspace.
        output_data_uri = ml_client.data.get(output_data.data_asset.name, version=output_data.data_asset.version).path
        return Output(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RW_MOUNT, path=output_data_uri)
    if output_data.blob_folder is None:
        # Output to workspace's default datastore.
        return Output(type=AssetTypes.CUSTOM_MODEL)
    else:
        raise "not implemented"
    
def submit_aml_job(
    experiment_name: str,
    display_name: str,
    aml_config: AmlConfig,
    cmd_line: str,
    input_data: Optional[list[NamedInputInfo]],
    output_data: Optional[list[NamedOutputInfo]],
    environment_name="spec_decoding",
    compute_cluster="a100x1",
    shared_memory="12g",
    docker_container_base: str="mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:latest",
    aml_env_path: str="AML/prompt_compression.yml",
    instance_count: int=1,
    args=None,
    tags: Optional[dict]=None,
    priority: Optional[str]=None,
    ):
    
    submission_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ml_client, aml_env, env_vars = aml_utils.get_default_ws_env(aml_config, aml_env_path, docker_container_base, experiment_name, compute_cluster, instance_count)

    # Input data.
    inputs = None
    if input_data is not None:
        inputs = {
            d.name: resolve_input(d, ml_client)
            for d in input_data
        }

    # # training_data_asset_path = ml_client.data.get("quarot_phi36_w4_lora_base_train_data_tier1_0_605219", version='latest').path
    # draft_model_save_asset_path = ml_client.data.get("spec_decoding_phi36_draft_models", version='latest').path
    # phi_data_asset_path = "azureml://subscriptions/{}/resourcegroups/{}/workspaces/{}/datastores/{}/paths//".format(
    #     args.subscription_id, args.resource_group, args.workspace, "spec_decoding")
    

    outputs = {
        f"output_sbom_{experiment_name}": 
        Output(
            type=AssetTypes.CUSTOM_MODEL,
            path=f"azureml://datastores/workspaceartifactstore/paths/{experiment_name}/{submission_time}",
        )
    }
    if output_data is not None:
        for d in output_data:
            outputs[d.name] = resolve_output(d, ml_client)
        


    job = command(
        inputs=inputs,
        outputs=outputs,
        code="./",
        command=cmd_line,
        environment=aml_env,
        environment_variables=env_vars,
        compute=compute_cluster,
        display_name=display_name,
        experiment_name=experiment_name,
        shm_size=shared_memory,
        tags=tags or {},
    )
    
    # Set job priority if specified
    if priority:
        job.priority = priority

    aml_utils.submit_aml_job(ml_client, job)


CLI(submit_aml_job, as_positional=False)