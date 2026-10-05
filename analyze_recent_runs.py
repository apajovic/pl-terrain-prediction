#!/usr/bin/env python3
"""
Analyze recent training runs (April 1, 2026) and reconstruct loss curves
from checkpoint progression and MLflow artifacts.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def get_recent_runs(after_time="2026-04-01 15:00:00"):
    """Get runs from April 1, 2026 15:00 onwards."""
    mlruns_dir = Path("./mlruns/0")
    runs = []
    
    after_dt = datetime.strptime(after_time, "%Y-%m-%d %H:%M:%S")
    
    for run_dir in mlruns_dir.iterdir():
        if run_dir.is_dir() and len(run_dir.name) == 32:
            mtime = os.path.getmtime(run_dir)
            mtime_dt = datetime.fromtimestamp(mtime)
            
            if mtime_dt >= after_dt:
                artifacts_dir = run_dir / "artifacts"
                model_name = "unknown"
                
                if artifacts_dir.exists():
                    metrics_dir = artifacts_dir / "metrics"
                    if metrics_dir.exists():
                        for txt_file in metrics_dir.glob("*.txt"):
                            if txt_file.name != "metrics.txt":
                                model_name = txt_file.stem
                                break
                
                runs.append({
                    "run_id": run_dir.name[:8],
                    "full_id": run_dir.name,
                    "timestamp": mtime_dt,
                    "model": model_name,
                    "path": run_dir
                })
    
    return sorted(runs, key=lambda x: x["timestamp"])

def get_checkpoint_timestamps_for_model(model_name):
    """Get checkpoint creation times for a specific model."""
    model_dir = Path(f"./output/{model_name}")
    
    checkpoints = {}
    if model_dir.exists():
        for pth_file in model_dir.glob("*__epoch*.pth"):
            try:
                epoch_str = pth_file.stem.split("__epoch")[-1]
                epoch = int(epoch_str)
                mtime = datetime.fromtimestamp(pth_file.stat().st_mtime)
                
                if epoch not in checkpoints:
                    checkpoints[epoch] = {
                        "path": str(pth_file),
                        "timestamp": mtime,
                        "size_mb": pth_file.stat().st_size / 1024**2
                    }
            except (ValueError, IndexError):
                continue
    
    return checkpoints

def analyze_recent_runs():
    """Detailed analysis of recent runs."""
    runs = get_recent_runs("2026-04-01 14:00:00")
    
    print("\n" + "="*80)
    print("RECENT TRAINING RUNS - April 1, 2026")
    print("="*80)
    
    # Group by model
    by_model = defaultdict(list)
    for run in runs:
        by_model[run["model"]].append(run)
    
    for model_name in sorted(by_model.keys()):
        model_runs = by_model[model_name]
        print(f"\n{model_name.upper()}: {len(model_runs)} runs")
        print("-" * 80)
        print("Run ID  | Time     | Checkpoints in output dir")
        print("-" * 80)
        
        checkpoints = get_checkpoint_timestamps_for_model(model_name)
        sorted_epochs = sorted(checkpoints.keys())
        
        for run in model_runs:
            print(f"{run['run_id']} | {run['timestamp'].strftime('%H:%M:%S')} | ", end="")
            
            # Find checkpoints saved during this run (roughly by timestamp)
            if run == model_runs[-1]:  # Last run
                # Show all remaining checkpoints
                print(f"epochs {min(sorted_epochs)}-{max(sorted_epochs)}")
            else:
                next_run = model_runs[model_runs.index(run) + 1]
                relevant = [e for e in sorted_epochs 
                           if checkpoints[e]["timestamp"] >= run["timestamp"] 
                           and checkpoints[e]["timestamp"] < next_run["timestamp"]]
                if relevant:
                    print(f"epochs {min(relevant)}-{max(relevant)}")
                else:
                    print("(no checkpoint saves in this window)")
        
        # Summary stats
        if checkpoints:
            print(f"\nCheckpoint progression for {model_name}:")
            print(f"  Total epochs trained: {max(sorted_epochs)}")
            print(f"  Unique epoch checkpoints saved: {len(sorted_epochs)}")
            print(f"  Epoch range: {min(sorted_epochs)}-{max(sorted_epochs)}")
            
            # Estimate best epoch based on where training stabilized
            # (best epochs are typically early, not late)
            early_epochs = [e for e in sorted_epochs if e <= 20]
            print(f"  Early epochs (≤20): {early_epochs}")

def plot_recent_training():
    """Create visualization of recent training runs."""
    runs = get_recent_runs("2026-04-01 14:00:00")
    
    # Get unique models
    models = sorted(set(r["model"] for r in runs if r["model"] != "unknown"))
    
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
    if len(models) == 1:
        axes = [axes]
    
    for ax, model_name in zip(axes, models):
        checkpoints = get_checkpoint_timestamps_for_model(model_name)
        
        if not checkpoints:
            ax.text(0.5, 0.5, f"No checkpoints for {model_name}", 
                   ha='center', va='center')
            ax.set_title(f"{model_name}")
            continue
        
        sorted_epochs = sorted(checkpoints.keys())
        timestamps = [checkpoints[e]["timestamp"] for e in sorted_epochs]
        
        # Convert to hours since first checkpoint
        start_time = min(timestamps)
        hours_since_start = [(t - start_time).total_seconds() / 3600 for t in timestamps]
        
        # Color points by run
        model_runs = [r for r in runs if r["model"] == model_name]
        colors = []
        for epoch, ts in zip(sorted_epochs, timestamps):
            # Find which run this checkpoint belongs to
            for i, run in enumerate(model_runs):
                if run["timestamp"] <= ts:
                    colors.append(i)
        
        scatter = ax.scatter(hours_since_start, sorted_epochs, c=colors, cmap='viridis', s=50, alpha=0.7)
        ax.plot(hours_since_start, sorted_epochs, alpha=0.3, linestyle='--', color='gray')
        
        ax.set_xlabel("Hours since first checkpoint")
        ax.set_ylabel("Epoch Number")
        ax.set_title(f"{model_name.upper()}: Recent Training\n({len(model_runs)} runs, {max(sorted_epochs)} epochs)")
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for runs
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Run #")
    
    plt.tight_layout()
    plt.savefig("recent_training_runs.png", dpi=150, bbox_inches='tight')
    print("\n✓ Saved: recent_training_runs.png")
    plt.show()

if __name__ == "__main__":
    analyze_recent_runs()
    print("\n\nGenerating visualization...")
    plot_recent_training()
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("""
The visualization shows checkpoint progression for recent runs (April 1).
Each point represents a saved checkpoint, colored by which training run it
belongs to (Run 1, Run 2, etc.).

Key observations:
- X-axis: Time progression (hours from first training)
- Y-axis: Epoch number being trained
- Each color: A separate model run (loading checkpoint from previous run)

If points jump backward (epoch-wise), it means a new run loaded a checkpoint
from the middle of a previous run's training (checkpoint resumption).
""")
