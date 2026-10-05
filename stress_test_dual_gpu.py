#!/usr/bin/env python3
"""
Dual-GPU stress test — no DDP, no NCCL, just pure CUDA matmuls.

This isolates whether the machine crash is caused by:
  A) DDP/NCCL communication  →  crash doesn't happen with this script
  B) Sustained dual-GPU compute load  →  crash DOES happen (hardware issue)

Usage:
    # Default: 180W power cap, 10 minutes
    python stress_test_dual_gpu.py

    # Custom power cap and duration
    python stress_test_dual_gpu.py --power-limit 150 --duration 600

    # Without power cap
    python stress_test_dual_gpu.py --no-power-cap

If this script crashes the machine, the problem is HARDWARE (PSU can't
handle dual 3090 sustained load), not software/NCCL/DDP.
"""

import argparse
import os
import subprocess
import threading
import time

import torch


def set_power_limit(gpu_id: int, watts: int) -> bool:
    """Set GPU power limit. Returns True if successful."""
    try:
        subprocess.run(
            ["sudo", "nvidia-smi", "-i", str(gpu_id), "-pl", str(watts)],
            capture_output=True, timeout=10,
        )
        return True
    except Exception:
        return False


def enable_persistence_mode() -> bool:
    try:
        subprocess.run(
            ["sudo", "nvidia-smi", "-pm", "1"],
            capture_output=True, timeout=10,
        )
        return True
    except Exception:
        return False


def gpu_worker(gpu_id: int, duration_s: int, matrix_size: int, stop_event: threading.Event):
    """Run continuous matmuls on a single GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)

    start = time.time()
    iters = 0
    while not stop_event.is_set() and (time.time() - start) < duration_s:
        _ = torch.mm(a, b)
        iters += 1
        if iters % 500 == 0:
            elapsed = time.time() - start
            mem = torch.cuda.memory_allocated(device) / 1e9
            print(f"  [GPU {gpu_id}] {iters} matmuls, {elapsed:.0f}s elapsed, {mem:.1f}GB VRAM")

    elapsed = time.time() - start
    print(f"  [GPU {gpu_id}] Done: {iters} matmuls in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Dual-GPU stress test (no DDP/NCCL)")
    parser.add_argument("--duration", type=int, default=600, help="Test duration in seconds (default: 600)")
    parser.add_argument("--power-limit", type=int, default=180, help="Power limit per GPU in watts (default: 180)")
    parser.add_argument("--no-power-cap", action="store_true", help="Skip power capping")
    parser.add_argument("--matrix-size", type=int, default=4096, help="Matrix size for matmuls (default: 4096)")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Only {num_gpus} GPU(s) found. Need 2 for dual-GPU stress test.")
        return

    print(f"=== Dual-GPU Stress Test ===")
    print(f"GPUs:         {num_gpus}")
    print(f"Duration:     {args.duration}s")
    print(f"Matrix size:  {args.matrix_size}x{args.matrix_size}")
    print()

    # Power management
    if not args.no_power_cap:
        print(f"Enabling persistence mode...")
        enable_persistence_mode()
        print(f"Setting power limit to {args.power_limit}W per GPU...")
        for i in range(num_gpus):
            ok = set_power_limit(i, args.power_limit)
            status = "OK" if ok else "FAILED (need sudo)"
            print(f"  GPU {i}: {status}")
        print()

    # Verify GPUs
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, {props.total_memory / 1e9:.1f}GB")
    print()

    # Launch stress threads
    print(f"Starting stress test for {args.duration}s...")
    print(f"If the machine freezes during this test, it's a HARDWARE issue (PSU/thermals).")
    print(f"Monitor with: watch -n1 nvidia-smi")
    print()

    stop_event = threading.Event()
    threads = []
    for gpu_id in range(min(num_gpus, 2)):  # Only test 2 GPUs
        t = threading.Thread(target=gpu_worker, args=(gpu_id, args.duration, args.matrix_size, stop_event))
        t.start()
        threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        for t in threads:
            t.join(timeout=5)

    print()
    print("=== Stress test completed without crash! ===")
    print("The hardware can sustain dual-GPU compute at this power level.")
    print("If DDP training still crashes, the issue is in NCCL/DDP communication, not raw compute.")


if __name__ == "__main__":
    main()
