import os
from typing import Any, Dict

import torch


def save_checkpoint(payload: Dict[str, Any], path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    staging = f"{path}.tmp"
    torch.save(payload, staging)
    os.replace(staging, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    loaded: Dict[str, Any] = torch.load(path, map_location="cpu")
    return loaded
