import os
from typing import Tuple

import torch
import torch.distributed as dist


def init_process_group() -> Tuple[int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world


def shutdown() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
