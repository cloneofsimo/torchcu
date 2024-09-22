from pathlib import Path
import runpy
import subprocess
import numpy as np
import torch
import ctypes
from typing import Callable

from rich.table import Table
from rich.console import Console


table = Table(title="Transpilation Results")
table.add_column("File", style="cyan", no_wrap=True)
table.add_column("Score", style="magenta")


CURR_DIR = Path(__file__).parent
DATA_DIR = CURR_DIR / "test_data"

def load_pytorch_function(file_path: str, function_name: str) -> Callable:
    """Load the PyTorch function from the given file."""
    module = runpy.run_path(file_path)
    return module[function_name]

def load_function_signature(file_path: str) -> tuple[str, list[tuple[str, tuple, torch.dtype]]]:
    """Load the function signature from the given file."""
    module = runpy.run_path(file_path)
    name, *args = module['function_signature']
    # validate the args follow the format
    # ('name', (shape,), dtype)

    for arg in args:
        if not isinstance(arg, tuple):
            raise ValueError(f"Invalid argument: {arg}")

        if len(arg) != 3:
            raise ValueError(f"Invalid argument: {arg}")

        if not isinstance(arg[0], str):
            raise ValueError(f"Invalid argument: {arg}")

        if not isinstance(arg[1], tuple):
            raise ValueError(f"Invalid argument: {arg}")
        if len(arg[1]) < 1:
            raise ValueError(f"Invalid argument: {arg}")
        if not all(isinstance(dim, int) for dim in arg[1]):
            raise ValueError(f"Invalid argument: {arg}")

        if not isinstance(arg[2], torch.dtype):
            raise ValueError(f"Invalid argument: {arg}")

    return name, args


def transpile_to_cuda(file_path: str) -> Path:
    """Transpile the PyTorch function to CUDA code."""
    output_file = Path(file_path).with_suffix('.cu')
    print(f"Transpiling to CUDA: {output_file}")
    return output_file

def compile_cuda(cuda_file: Path) -> Path:
    """Compile the CUDA code to a shared library."""
    output_file = cuda_file.with_suffix('.so')
    subprocess.run(['nvcc', '-Xcompiler', '-fPIC', '--shared', '-o', str(output_file), str(cuda_file)], check=True)
    return output_file

def prepare_inputs(signature: list) -> list:
    """Prepare input tensors based on the function signature."""
    inputs = {}
    for arg in signature:
        if isinstance(arg, tuple):
            inputs[arg[0]] = torch.randn(arg[1], dtype=arg[2])
        else:
            inputs[arg] = torch.randn(1, dtype=torch.float32)
    return inputs

def run_pytorch_function(func: Callable, inputs: dict) -> torch.Tensor:
    """Run the PyTorch function with the given inputs."""
    with torch.no_grad():
        return func(**inputs)


def load_cuda_function(lib_path: Path, function_name: str, signature: list) -> Callable:
    lib_path = lib_path.resolve()
    lib = ctypes.CDLL(lib_path)
    func = getattr(lib, function_name)
    args = [
        ctypes.c_int # num_args
    ]

    # ('input_tensor', (4, 4), torch.float32)
    # ('weight', (4, 4), torch.float32)
    # ('other', (2, 4, 4), torch.float32)
    for arg in signature:
        arg_name, arg_shape, arg_dtype = arg
        
        args.append(ctypes.POINTER(ctypes.c_float))
        for dim in arg_shape:
            args.append(ctypes.c_int)

    func.argtypes = args
    func.restype = None
    return func

def run_cuda_function(func: Callable, inputs: dict, output_shape: tuple) -> np.ndarray:
    input_tensor = inputs['input_tensor'].cpu().numpy()
    weight = inputs['weight'].cpu().numpy()
    output = np.zeros(output_shape, dtype=np.float32)

    func(
        ctypes.c_int(7),  # num_args (excluding this one)
        input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(input_tensor.shape[0]),
        ctypes.c_int(input_tensor.shape[1]),
        weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(weight.shape[0]),
        ctypes.c_int(weight.shape[1]),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )

    return output

def compare_outputs(pytorch_output: torch.Tensor, cuda_output: np.ndarray, rtol: float = 1e-2, atol: float = 1e-2) -> bool:
    """Compare the outputs of PyTorch and CUDA implementations."""
    max_diff = np.max(np.abs(pytorch_output.cpu().numpy() - cuda_output))
    allclose = torch.allclose(pytorch_output.cpu(), torch.tensor(cuda_output), rtol=rtol, atol=atol)
    return allclose, max_diff

def judge_transpilation(input_file: Path, num_tests: int = 10) -> None:
    """Judge the correctness of the transpiled CUDA code."""
    print(f"Testing transpilation of {input_file}")

    # Load PyTorch function and signature
    function_name, signature = load_function_signature(input_file)
    pytorch_func = load_pytorch_function(input_file, function_name)

    print(f"Loaded PyTorch function: {function_name} with signature {signature}")

    # Transpile and compile CUDA code
    cuda_file = transpile_to_cuda(input_file)
    lib_path = compile_cuda(cuda_file)
    cuda_func = load_cuda_function(lib_path, function_name, signature)

    passed_tests = 0
    total_diff = 0

    for i in range(num_tests):
        # Prepare inputs
        inputs = prepare_inputs(signature)

        # Run PyTorch function
        pytorch_output = run_pytorch_function(pytorch_func, inputs)

        # Run CUDA function
        cuda_output = run_cuda_function(cuda_func, inputs, pytorch_output.shape)

        # Compare outputs
        passed, max_diff = compare_outputs(pytorch_output, cuda_output)
        total_diff += max_diff

        passed_tests += passed

    return passed_tests / num_tests

if __name__ == "__main__":
    for file in DATA_DIR.glob("*.py"):
        try:
            score = judge_transpilation(file)
            table.add_row(file.name, f"{score * 100:.2f}%")
        except Exception as e:
            table.add_row(file.name, "0.0%")

    console = Console()
    console.print(table)

