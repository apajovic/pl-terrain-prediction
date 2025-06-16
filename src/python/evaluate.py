# evaluate.py
# Model evaluation script
from models import unet  # Add more models as needed
from data.dataloader import get_dataloaders
from config import get_config
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from postprocess import postprocess
import mlflow

# TODO: Add more model imports as implemented

def evaluate():
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet.UNet(config).to(device)
    # Load weights
    weights_path = config.get('evaluation.model_path', './best_model.pth')
    path_from_mlflow = config.get('evaluation.path_from_mlflow', False)
    if path_from_mlflow:
        print(f"Loading model weights from MLflow artifact: {weights_path}")
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=weights_path)
        model.load_state_dict(torch.load(local_path, map_location=device))
        print(f"Loaded weights from MLflow artifact at {weights_path}")
    elif os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Weights not found at {weights_path}, using random init.")
    _, test_loader = get_dataloaders(config)
    model.eval()
    preds = []
    targets = []
    total_pred_time = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, targs in test_loader:
            inputs = inputs.to(device)
            targs = targs.to(device)
            start_time = time.perf_counter()
            outputs = model(inputs)
            end_time = time.perf_counter()
            batch_time = end_time - start_time
            total_pred_time += batch_time
            num_batches += 1
            preds.append(outputs.cpu())
            targets.append(targs.cpu())
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    # Compute MSE
    mse = torch.nn.functional.mse_loss(preds, targets).item()
    print(f"Test MSE: {mse:.6f}")

    avg_pred_time = total_pred_time / num_batches if num_batches > 0 else 0.0
    print(f"Total prediction time: {total_pred_time:.4f} s")
    print(f"Average prediction time per batch: {avg_pred_time:.6f} s")

    mlflow.start_run(run_name="evaluate")
    mlflow.log_params({
        'model': config.get('model.name', 'unet'),
        'channels': config.get('model.params.channels', 16),
        'layers': config.get('model.params.layers', 7)
    })
    mlflow.log_metrics({
        'test_mse': mse,
        'total_pred_time': total_pred_time,
        'avg_pred_time_per_batch': avg_pred_time
    })

    # Postprocess and plot/save predictions
    postprocess(preds, config, save_dir=config.get('output.wrap_pred_dir'), show=True)
    # Optionally: save metrics
    metrics_path = config.get('output.metrics_out', './metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Test MSE: {mse:.6f}\n")
        f.write(f"Total prediction time: {total_pred_time:.4f} s\n")
        f.write(f"Average prediction time per batch: {avg_pred_time:.6f} s\n")
    print(f"Metrics saved to {metrics_path}")
    if os.path.exists(metrics_path):
        mlflow.log_artifact(metrics_path, artifact_path="metrics")
    mlflow.end_run()

if __name__ == '__main__':
    evaluate()
