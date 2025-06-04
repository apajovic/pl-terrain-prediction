# evaluate.py
# Model evaluation script
from models import unet  # Add more models as needed
from data.dataloader import get_dataloader
from config import get_config
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from postprocess import postprocess

# TODO: Add more model imports as implemented

def evaluate():
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet.UNet(config).to(device)
    # Load weights
    weights_path = config.get('model.model_out', './best_model.pth')
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Weights not found at {weights_path}, using random init.")
    test_loader = get_dataloader(config, train=False)
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for inputs, targs in test_loader:
            inputs = inputs.to(device)
            targs = targs.to(device)
            outputs = model(inputs)
            preds.append(outputs.cpu())
            targets.append(targs.cpu())
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    # Compute MSE
    mse = torch.nn.functional.mse_loss(preds, targets).item()
    print(f"Test MSE: {mse:.6f}")
    # Postprocess and plot/save predictions
    postprocess(preds, config, save_dir=config.get('model.wrap_pred_dir'), show=True)
    # Optionally: save metrics
    metrics_path = config.get('output.metrics_out', './metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Test MSE: {mse:.6f}\n")
    print(f"Metrics saved to {metrics_path}")

if __name__ == '__main__':
    evaluate()
