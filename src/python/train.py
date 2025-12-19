# train.py
# Main training script (configurable, supports Optuna, local/AML)
import os
import torch
import torch.distributed as dist
import mlflow

from data.dataloader import get_dataloaders, ROW_SPLIT_TRANSFORM, BASIC_TRANSFORM
from ddp_utils import (
    initialize_ddp,
    cleanup_ddp,
    get_rank,
    get_world_size,
    is_main_process,
    is_distributed_training,
    synchronize,
    reduce_tensor,
)
import matplotlib.pyplot as plt
from config import get_config
from utils import set_seed
import optuna
from preprocess import unwrap_img
from postprocess import wrap_img
from metrics import rmse_to_dB
from models import get_model
import tqdm
import numpy as np
import time


def preprocess_data(config):
    unwrap_img(
        config.get("data.wrap_dir"),
        config.get("data.unwrap_dir"),
        config.get("data.num_angles", 256),
        config.get("data.num_radii", 256),
        (256, 256),
        config.get("data.base_name", "PL"),
    )


def postprocess_and_plot(pred_tensor, config, save_dir=None, show=True):
    pred_np = pred_tensor.detach().cpu().numpy().squeeze(1)
    wrap_dir = save_dir or config.get("output.wrap_pred_dir", "./wrap_pred")
    os.makedirs(wrap_dir, exist_ok=True)
    wrapped = wrap_img(
        wrap_dir,
        pred_np,
        (256, 256),
        config.get("data.base_name", "PL_pred"),
        indikator=True,
    )
    for i in range(min(4, wrapped.shape[-1])):
        plt.imshow(wrapped[:, :, i], cmap="gray")
        plt.title(f"Postprocessed {i+1}")
        if save_dir:
            plt.savefig(os.path.join(wrap_dir, f"postprocessed_{i+1}.png"))
            mlflow.log_artifact(
                os.path.join(wrap_dir, f"postprocessed_{i+1}.png"),
                artifact_path="output_images",
            )
        if show:
            plt.show()
        plt.close()
    return wrapped


def train_one_epoch(model, loader, criterion, optimizer, device, epoch=0):
    model.train()
    running_loss = 0.0
    
    # Set epoch for DistributedSampler to ensure proper shuffling across ranks
    if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    
    # Only show progress bar on main process
    progress_bar = None
    if is_main_process():
        progress_bar = tqdm.tqdm(
            loader, desc="Training", total=len(loader), leave=False, dynamic_ncols=True
        )
    else:
        progress_bar = loader
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if is_main_process():
            progress_bar.set_postfix(loss=loss.item())
    
    if is_main_process():
        progress_bar.close()
    
    # Synchronize loss across all processes
    avg_loss = running_loss / len(loader)
    if is_distributed_training():
        loss_tensor = torch.tensor([avg_loss], dtype=torch.float32, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor.item() / get_world_size())
    
    return avg_loss


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    preds = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            # Keep predictions on GPU for DDP gathering
            preds.append(outputs)
    
    # Concatenate predictions (still on GPU if distributed)
    preds = torch.cat(preds, dim=0) if preds else torch.tensor([]).to(device)
    
    # Synchronize validation loss across all processes
    avg_loss = val_loss / len(loader)
    if is_distributed_training():
        loss_tensor = torch.tensor([avg_loss], dtype=torch.float32, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor.item() / get_world_size())
        
        # Gather all predictions on rank 0 (keep on GPU during gather)
        world_size = get_world_size()
        if get_rank() == 0:
            gathered_preds = [torch.zeros_like(preds) for _ in range(world_size)]
            dist.gather(preds, gathered_preds, dst=0)
            # Concatenate and move to CPU after gathering
            preds = torch.cat(gathered_preds, dim=0).cpu()
        else:
            dist.gather(preds, None, dst=0)
            # Non-main processes can clear predictions to save memory
            preds = torch.tensor([])
    else:
        # Move to CPU for single GPU case
        preds = preds.cpu()
    
    return avg_loss, preds


def objective(trial, num_epochs=10):
    config = get_config()
    set_seed(config.get("training.seed", 42))

    # trial suggestions for hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    # batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    channels = trial.suggest_categorical("channels", [8, 16, 32, 64])
    layers = trial.suggest_categorical("layers", [3, 5, 7])

    config.config["model"]["params"]["channels"] = channels
    config.config["model"]["params"]["layers"] = layers

    device_optuna = torch.device(config.get("training.device", "cpu"))
    model = get_model(config).to(device_optuna)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    two_channel_split = config.get("data.split_channels", False)

    transform = ROW_SPLIT_TRANSFORM if two_channel_split else BASIC_TRANSFORM
    train_loader, val_loader = get_dataloaders(
        config, val_split=0.1, transform=transform
    )
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(num_epochs):
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device_optuna,
        )

    val_loss, _ = validate(
        model, val_loader, criterion, device_optuna
    )
    return val_loss


def main(config):
    # Initialize distributed training if available
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        initialize_ddp()
    
    # Print config only on main process
    if is_main_process():
        print("Loaded config:", config.as_dict())
    
    set_seed(config.get("training.seed", 42))
    
    # Set device based on DDP or single GPU
    if is_distributed_training():
        device = torch.device(f"cuda:{get_rank()}")
    else:
        device = torch.device(config.get("training.device", "cpu"))
    
    config.config["device"] = device
    
    if is_main_process():
        if not config.get("data", {}).get("skip_preprocess", False):
            print("Starting preprocessing...")
            preprocess_data(config)
            print("Preprocessing done.")
        else:
            print("Skipping preprocessing.")
    
    # Synchronize after preprocessing
    if is_distributed_training():
        synchronize()
    
    two_channel_split = config.get("data.split_channels", False)
    transform = ROW_SPLIT_TRANSFORM if two_channel_split else BASIC_TRANSFORM
    
    train_loader, val_loader = get_dataloaders(
        config, val_split=0.1, transform=transform, use_ddp=is_distributed_training()
    )
    
    if is_main_process():
        print("Data loaders ready.")

    model = get_model(config).to(device)
    
    # Wrap model in DistributedDataParallel if distributed training
    if is_distributed_training():
        if is_main_process():
            print(f"Using {get_world_size()} GPUs for distributed training")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[get_rank()], output_device=get_rank()
        )
    else:
        if is_main_process():
            print("Using single GPU/CPU")
    
    if is_main_process():
        print(f"Model '{config.get('model.name', 'unet')}' initialized.")

    # Load checkpoint if available and not training from scratch
    model_path = config.get("training.model_out", "./best_model.pth")
    if not config.get("training.from_scratch", False) and os.path.exists(model_path):
        if is_main_process():
            print(f"Loading model weights from checkpoint: {model_path}")
        
        # In DDP, load to CPU first to avoid device mismatch issues
        # Each process only sees its own GPU, so direct GPU loading fails
        map_location = 'cpu' if is_distributed_training() else device
        checkpoint = torch.load(model_path, map_location=map_location)
        
        # Handle loading checkpoints from both DDP and single-GPU models
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError:
            # If model is wrapped in DDP, adapt checkpoint keys
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                new_state_dict = {f'module.{k}': v for k, v in checkpoint.items() if not k.startswith('module.')}
                if new_state_dict:
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(checkpoint)
            else:
                raise
        
        if is_main_process():
            print("Checkpoint loaded successfully.")
    else:
        if is_main_process():
            print("Training from scratch (no checkpoint loaded).")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("training.lr", 1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get("training.lr_step", 80),
        gamma=config.get("training.lr_gamma", 0.1),
    )
    num_epochs = config.get("training.epochs", 150)
    best_val_loss = float("inf")
    best_preds = None

    # Only start MLflow run on main process
    if is_main_process():
        mlflow.start_run(run_name=f"train-{config.get('model.name')}-{int(time.time())}", tags={"model": config.get("model.name"), "data": config.get("data.input_dir")})
        mlflow.log_params(
            {
                "lr": config.get("training.lr", 1e-4),
                "batch_size": config.get("training.batch_size", 16),
                "epochs": num_epochs,
                "model": config.get("model.name", "unet"),
                "channels": config.get("model.params.channels", 16),
                "layers": config.get("model.params.layers", 7),
                "num_gpus": get_world_size(),
            }
        )

    for epoch in range(num_epochs):
        if is_main_process():
            print(f"Epoch {epoch+1}/{num_epochs} starting...")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch)
        val_loss, preds = validate(model, val_loader, criterion, device)
        scheduler.step()

        if is_main_process():
            print(
                f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"RMSE in DB: {rmse_to_dB(np.sqrt(val_loss)):.6f}"
            )
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_preds = preds
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                # Save properly whether model is wrapped in DDP or not
                model_state = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()
                torch.save(model_state, model_path)
                torch.save(model_state, model_path.replace('.pth', f'_epoch{epoch+1}.pth'))
                mlflow.log_artifact(model_path, artifact_path="models")
                print(
                    f"New best model saved at epoch {epoch+1} with val loss {val_loss:.4f}"
                )
        

    if best_preds is not None and is_main_process():
        print("Postprocessing and plotting best predictions...")
        postprocess_and_plot(
            best_preds, config, save_dir=config.get("output.wrap_pred_dir"), show=True
        )
        print("Postprocessing done.")
        # Optionally log output images/metrics
        metrics_path = config.get("output.metrics_out", "./metrics.txt")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        if os.path.exists(metrics_path):
            mlflow.log_artifact(metrics_path, artifact_path="metrics")
    
    # End MLflow run on main process
    if is_main_process():
        mlflow.end_run()
    
    # Clean up distributed training
    if is_distributed_training():
        cleanup_ddp()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optuna", action="store_true", help="Enable Optuna hyperparameter search"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="default_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Ignore checkpoint and train from scratch",
    )
    args = parser.parse_args()
    config = get_config(args.config)
    # Add from_scratch flag to config
    config.config["training.from_scratch"] = args.from_scratch
    if args.optuna:
        print("Optuna hyperparameter search enabled.")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=config.get("training.n_trials", 10))
        print("Best trial:", study.best_trial.params)
    else:
        main(config)
