# Get AML credential for compliance
import os
import sys
from pathlib import Path
from azure.identity import InteractiveBrowserCredential, AuthenticationRecord, TokenCachePersistenceOptions
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from typing import Any
from azure.identity import DefaultAzureCredential
import webbrowser


def get_credential():
    """Retrieves a credential that can be used in Azure.

    The cached credential is silently retrieved (if available), otherwise the user is prompted to perform
    interactive authentication and the new credential is cached for future use.
    
    Supports multiple auth methods:
    - Cached browser credential (if available)
    - Device code flow (WSL, SSH, headless environments)
    - DefaultAzureCredential fallback (az login, environment variables)
    """
    from azure.identity import DeviceCodeCredential
    
    lib_name = "aml_training"

    if sys.platform.startswith("win"):
        auth_record_root_path = Path(os.environ["LOCALAPPDATA"])
    else:
        auth_record_root_path = Path("~").expanduser()
    auth_record_path = auth_record_root_path / lib_name / "auth_record.json"
    cache_options = TokenCachePersistenceOptions(
        name=f"{lib_name}.cache", 
        allow_unencrypted_storage=True
    )
    
    # Try to use cached browser credential first
    if auth_record_path.exists():
        try:
            with open(auth_record_path, "r") as f:
                record_json = f.read()
            deserialized_record = AuthenticationRecord.deserialize(record_json)
            credential = InteractiveBrowserCredential(
                authentication_record=deserialized_record,
                cache_persistence_options=cache_options
            )
            # Verify the credential works
            credential.get_token("https://management.azure.com/.default")
            return credential
        except Exception as e:
            print(f"Cached credential failed ({type(e).__name__}), trying device code flow...")
    
    # Detect if running in headless environment (WSL, SSH, containers)
    is_headless = "WSL_DISTRO_NAME" in os.environ or \
                  "WSL_INTEROP" in os.environ or \
                  os.environ.get("TERM") == "xterm" or \
                  not os.environ.get("DISPLAY")
    
    if is_headless:
        print("\n" + "="*60)
        print("Headless environment detected (WSL/SSH).")
        print("Using device code flow for authentication.")
        print("="*60)
        print("\nYou will see:")
        print("  1. A URL to visit (e.g., https://microsoft.com/devicelogin)")
        print("  2. A device code to enter at that URL")
        print("\nFollow the prompts below:\n")
        return DeviceCodeCredential()
    
    # Try browser credential for interactive environments
    try:
        auth_record_path.parent.mkdir(parents=True, exist_ok=True)
        credential = InteractiveBrowserCredential(
            cache_persistence_options=cache_options
        )
        record_json = credential.authenticate().serialize()
        with open(auth_record_path, "w") as f:
            f.write(record_json)
        return credential
    except Exception as e:
        print(f"Browser authentication failed ({type(e).__name__}), "
              f"falling back to device code flow...")
        return DeviceCodeCredential()


def get_default_ws_env(
        aml_config,
        aml_env_path,
        docker_container_base,
        experiment_name,
        compute_cluster_name,
        instance_count,
):
    """
    Get handle to the workspace, setup enviornment and environment variables
    Returns:
        workspace handler, custom environment and predefined env variables
    """
    # access_credential = DefaultAzureCredential()
    access_credential = get_credential()
    SUBSCRIPTION = aml_config.subscription_id
    RESOURCE_GROUP = aml_config.resource_group_name
    WS_NAME = aml_config.workspace_name
    print(f"Workspace Config:\nSUBSCRIPTION: {SUBSCRIPTION}"
          f"\nRESOURCE_GROUP: {RESOURCE_GROUP}"
          f"\nWS_NAME: {WS_NAME}")

    # Get a handle to the workspace
    ml_client = MLClient(
        credential=access_credential,
        subscription_id=SUBSCRIPTION,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WS_NAME,
    )


    # Verify that the handle works correctly.
    ws = ml_client.workspaces.get(WS_NAME)
    print(f"{ws.location} : {ws.resource_group}")

    # Setup environment
    aml_env_name = os.path.splitext(os.path.basename(aml_env_path))[0]
    custom_job_env = Environment(
        name=aml_env_name,
        description=f"Custom environment [{aml_env_name}] for AML training.",
        conda_file=aml_env_path,
        image=docker_container_base
    )
    custom_job_env = ml_client.environments.create_or_update(custom_job_env)
    print(f"Environment name: {aml_env_name}, conda_file: {aml_env_path}, docker_base_image: {docker_container_base}")

    # Turn on SBOM generation and some environment variables
    environment_variables = {
        "AZUREML_COMMON_RUNTIME_USE_SBOM_CAPABILITY": "true",
        "DATASET_MOUNT_FILE_CACHE_PRUNE_THRESHOLD": "0.7",
        "DATASET_MOUNT_BLOCK_BASED_CACHE_ENABLED": "true",
        "DATASET_MOUNT_BLOCK_FILE_CACHE_ENABLED": "false",
        "LOCAL_WORLD_SIZE": compute_cluster_name[-1],
        "WORLD_SIZE": str(int(compute_cluster_name[-1]) * instance_count),
        "NUM_CANONICAL_NODES": str(instance_count),
        "AZUREML_EXPERIMENT_NAME": experiment_name,
    }
    print(f"Default environment variables: {environment_variables}")

    return ml_client, custom_job_env, environment_variables


def submit_aml_job(ml_client: Any, job_cmd: Any):
    """AML job submit func.
    Args:
        ml_client   : handle to the workspace
        job_cmd     : job command
    Returns:
        AML studio url webbrowser
    """
    pipeline_job = ml_client.jobs.create_or_update(job_cmd)
    print(f"\nAML Studio URL: {pipeline_job.studio_url}")
    print("\nOpening in browser...")
    try:
        webbrowser.open(pipeline_job.studio_url)
    except Exception as e:
        print(f"Could not open browser ({type(e).__name__}).")
        print(f"Please visit manually: {pipeline_job.studio_url}")
