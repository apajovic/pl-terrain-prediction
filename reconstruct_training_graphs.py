#!/usr/bin/env python3
"""
Reconstruct train/val loss graphs from MLflow runs and checkpoint timestamps.
Shows how each model evolved across the checkpoint resumption iterations.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def get_checkpoint_info(model_name, output_dir="./output"):
    """Extract checkpoint epoch numbers and creation times."""
    model_dir = Path(output_dir) / model_name
    
    checkpoints = {}
    if model_dir.exists():
        for pth_file in model_dir.glob("*__epoch*.pth"):
            # Extract epoch number from filename
            try:
                epoch_str = pth_file.stem.split("__epoch")[-1]
                epoch = int(epoch_str)
                # Get file modification time
                mtime = datetime.fromtimestamp(pth_file.stat().st_mtime)
                if epoch not in checkpoints:
                    checkpoints[epoch] = mtime
                else:
                    # Keep the earliest timestamp for this epoch
                    if mtime < checkpoints[epoch]:
                        checkpoints[epoch] = mtime
            except (ValueError, IndexError):
                continue
    
    return checkpoints

def get_mlflow_runs(mlruns_dir="./mlruns/0"):
    """Get all MLflow run directories with their timestamps."""
    runs = []
    mlruns_path = Path(mlruns_dir)
    
    if mlruns_path.exists():
        for run_dir in mlruns_path.iterdir():
            if run_dir.is_dir() and len(run_dir.name) == 32:  # MLflow run ID format
                artifacts_dir = run_dir / "artifacts"
                if artifacts_dir.exists():
                    # Get run creation time from directory
                    mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
                    
                    # Try to extract model name from artifacts
                    model_name = None
                    metrics_dir = artifacts_dir / "metrics"
                    if metrics_dir.exists():
                        for txt_file in metrics_dir.glob("*.txt"):
                            model_name = txt_file.stem
                            break
                    
                    if model_name:
                        runs.append({
                            "run_id": run_dir.name[:8],
                            "model": model_name,
                            "timestamp": mtime,
                            "path": str(run_dir)
                        })
    
    return sorted(runs, key=lambda x: (x["model"], x["timestamp"]))

def analyze_checkpoint_progression(model_name):
    """Analyze how epochs progressed across runs (checkpoint resumption pattern)."""
    checkpoints = get_checkpoint_info(model_name)
    
    if not checkpoints:
        print(f"No checkpoints found for {model_name}")
        return None
    
    sorted_epochs = sorted(checkpoints.keys())
    print(f"\n{model_name.upper()}")
    print(f"  Epochs trained: {min(sorted_epochs)}-{max(sorted_epochs)}")
    print(f"  Total unique epochs: {len(sorted_epochs)}")
    
    # Detect runs (gaps or jumps in epoch sequence)
    runs = []
    current_run = [sorted_epochs[0]]
    
    for i in range(1, len(sorted_epochs)):
        # If gap > 1 or timestamp jump, new run started
        if sorted_epochs[i] - sorted_epochs[i-1] > 1:
            runs.append(current_run)
            current_run = [sorted_epochs[i]]
        else:
            current_run.append(sorted_epochs[i])
    
    runs.append(current_run)
    
    print(f"\n  Detected {len(runs)} separate runs:")
    for i, run in enumerate(runs, 1):
        start_ep = min(run)
        end_ep = max(run)
        num_eps = len(run)
        time_range = f"{checkpoints[start_ep].strftime('%Y-%m-%d')} to {checkpoints[end_ep].strftime('%Y-%m-%d')}"
        print(f"    Run {i}: epochs {start_ep}-{end_ep} ({num_eps} saved) - {time_range}")
    
    return {
        "epochs": sorted_epochs,
        "timestamps": [checkpoints[e] for e in sorted_epochs],
        "runs": runs
    }

def plot_checkpoint_progression(models=["radiounet", "pmnet_v3", "mamba"]):
    """Create visualization of checkpoint progression across runs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, model_name in zip(axes, models):
        data = analyze_checkpoint_progression(model_name)
        
        if data is None:
            continue
        
        epochs = data["epochs"]
        timestamps = data["timestamps"]
        
        # Convert timestamps to days since first epoch
        if timestamps:
            start_time = min(timestamps)
            days_since_start = [(t - start_time).days for t in timestamps]
            
            ax.scatter(days_since_start, epochs, alpha=0.6, s=30)
            ax.plot(days_since_start, epochs, alpha=0.3, linestyle='--')
            
            ax.set_xlabel("Days since first training")
            ax.set_ylabel("Epoch Number")
            ax.set_title(f"{model_name.upper()}: Checkpoint Progression")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("checkpoint_progression.png", dpi=150, bbox_inches='tight')
    print("\n✓ Saved: checkpoint_progression.png")
    plt.show()

def summarize_training_history():
    """Print summary of all training runs."""
    print("\n" + "="*70)
    print("MLFLOW RUNS SUMMARY")
    print("="*70)
    
    runs = get_mlflow_runs()
    
    models_runs = defaultdict(list)
    for run in runs:
        models_runs[run["model"]].append(run)
    
    for model, model_runs in sorted(models_runs.items()):
        print(f"\n{model.upper()}: {len(model_runs)} runs")
        for i, run in enumerate(model_runs, 1):
            print(f"  Run {i}: {run['run_id']} - {run['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRAINING PROGRESSION ANALYSIS")
    print("="*70)
    
    # Analyze checkpoint progression
    for model in ["radiounet", "pmnet_v3", "mamba"]:
        analyze_checkpoint_progression(model)
    
    # Summary of MLflow runs
    summarize_training_history()
    
    # Create visualization
    print("\nGenerating checkpoint progression plot...")
    plot_checkpoint_progression()
    
    print("\n" + "="*70)
    print("Key Insight:")
    print("  The checkpoint progression shows how many times each model was")
    print("  retrained (checkpoint resumption). Gaps in epoch numbers indicate")
    print("  separate training runs loading the best checkpoint from the previous run.")
    print("="*70)
