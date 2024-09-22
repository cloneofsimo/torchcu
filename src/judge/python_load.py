import runpy
from typing import Callable

import torch


def load_pytorch_function(file_path: str, function_name: str) -> Callable:
    """Load the PyTorch function from the given file."""
    module = runpy.run_path(file_path)
    return module[function_name]


def load_function_signature(
    file_path: str,
) -> tuple[str, list[tuple[str, tuple, torch.dtype]]]:
    """Load the function signature from the given file."""
    module = runpy.run_path(file_path)
    sig = module["function_signature"]
    name = sig["name"]
    args = sig["inputs"]
    # validate the args follow the format
    # [((shape,), dtype)]

    for arg in args:
        if not isinstance(arg, tuple):
            raise ValueError(f"Invalid input signature: {arg}")

        if len(arg) != 2:
            raise ValueError(f"Invalid input signature: {arg}")

        if not isinstance(arg[0], tuple):
            raise ValueError(f"Invalid input shape: {arg}")
        if len(arg[0]) < 1:
            raise ValueError(f"Invalid input shape: {arg}")
        if not all(isinstance(dim, int) for dim in arg[0]):
            raise ValueError(f"Invalid input shape: {arg}")

        if not isinstance(arg[1], torch.dtype):
            raise ValueError(f"Invalid input dtype: {arg}")

    return name, args
