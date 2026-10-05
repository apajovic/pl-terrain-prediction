# ddp_utils.py
# Utility functions for distributed data parallel training

import os
import torch
import torch.distributed as dist
from typing import Optional


def initialize_ddp(backend: str = None) -> None:
    """Initialize the distributed environment using torchrun env vars.
    
    Args:
        backend: 'nccl' or 'gloo'. If None, reads DDP_BACKEND env var
                 (default: 'gloo' for safety on this hardware).
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if dist.is_initialized():
        return

    if backend is None:
        backend = os.environ.get("DDP_BACKEND", "gloo")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Pin this process to its GPU BEFORE init_process_group so that
    # NCCL binds to the correct device from the start.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # For nccl: pass device_id so NCCL knows exactly which GPU this rank owns.
    # For gloo: device_id is not supported — gloo does CPU-side collectives.
    init_kwargs = dict(
        backend=backend,
        init_method="env://",
    )
    if backend == "nccl":
        init_kwargs["device_id"] = torch.device("cuda", local_rank)

    dist.init_process_group(**init_kwargs)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(
        f"[Rank {rank}] DDP ready  |  backend={backend}  world_size={world_size}  "
        f"local_rank={local_rank}  device=cuda:{torch.cuda.current_device()}"
    )


def cleanup_ddp() -> None:
    """Destroy the process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def is_distributed_training() -> bool:
    return dist.is_available() and dist.is_initialized()


def synchronize() -> None:
    """Barrier across all ranks."""
    if is_distributed_training():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, world_size: Optional[int] = None) -> torch.Tensor:
    """All-reduce tensor (mean) across ranks. No-op when world_size == 1."""
    if world_size is None:
        world_size = get_world_size()
    if world_size == 1 or not is_distributed_training():
        return tensor
    t = tensor.clone().detach()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t.div_(world_size)
    return t
