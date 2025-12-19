# ddp_utils.py
# Utility functions for distributed data parallel training

import os
import torch
import torch.distributed as dist
from typing import Optional


def initialize_ddp(backend: str = "nccl") -> None:
    """
    Initialize the distributed environment.
    
    Args:
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU/mixed)
    
    Raises:
        RuntimeError: If distributed environment is not properly set up
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    
    if dist.is_initialized():
        return
    
    # Environment variables should be set by torchrun/torch.distributed.launch
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    
    # Set CUDA_VISIBLE_DEVICES if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    
    # Set device
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    
    print(f"[Rank {rank}] Initialized DDP with world_size={world_size}")


def cleanup_ddp() -> None:
    """Clean up distributed environment."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        print(f"[Rank {get_rank()}] Cleaned up DDP")


def get_rank() -> int:
    """Get the rank of the current process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def get_local_rank() -> int:
    """Get the local rank (GPU index on this node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def synchronize() -> None:
    """Synchronize all processes."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, world_size: Optional[int] = None) -> torch.Tensor:
    """
    Reduce tensor across all processes (sum).
    
    Args:
        tensor: Tensor to reduce
        world_size: Number of processes (auto-detected if None)
    
    Returns:
        Reduced tensor on all processes
    """
    if world_size is None:
        world_size = get_world_size()
    
    if world_size == 1:
        return tensor
    
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    
    tensor = tensor.clone().detach()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(world_size)
    return tensor


def is_distributed_training() -> bool:
    """Check if distributed training is enabled."""
    return dist.is_available() and dist.is_initialized()
