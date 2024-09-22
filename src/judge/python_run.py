from typing import Callable

import torch

from src.judge.python_load import load_function_signature, load_pytorch_function


def prepare_inputs(signature: list[tuple[tuple, torch.dtype]]) -> list:
    """Prepare input tensors based on the function signature."""
    inputs = []
    for arg in signature:
        inputs.append(torch.randn(arg[0], dtype=arg[1]))
    return inputs


def run_pytorch_function(func: Callable, inputs: list) -> torch.Tensor:
    """Run the PyTorch function with the given inputs."""
    with torch.no_grad():
        return func(*inputs)


def run_pytorch_file(input_file: str) -> torch.Tensor:
    print(f"Testing transpilation of {input_file}")

    function_name, signature = load_function_signature(input_file)
    pytorch_func = load_pytorch_function(input_file, function_name)

    inputs = prepare_inputs(signature)
    pytorch_output = run_pytorch_function(pytorch_func, inputs)

    return pytorch_output
