import runpy
import subprocess
import numpy as np
import torch
import ctypes
from typing import Callable
import os

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

def transpile_to_cuda(file_path: str) -> str:
    """Transpile the PyTorch function to CUDA code."""
    output_file = file_path.replace('.py', '.cu')
    return output_file

def compile_cuda(cuda_file: str) -> str:
    """Compile the CUDA code to a shared library."""
    output_file = cuda_file.replace('.cu', '.so')
    subprocess.run(['nvcc', '-Xcompiler', '-fPIC', '--shared', '-o', output_file, cuda_file], check=True)
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


def load_cuda_function(lib_path: str, function_name: str, signature: list) -> Callable:
    lib_path = os.path.abspath(lib_path)
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

def judge_transpilation(input_file: str, num_tests: int = 10) -> None:
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
        print(f"Running test {i+1} with inputs:\n{inputs}")

        # Run PyTorch function
        pytorch_output = run_pytorch_function(pytorch_func, inputs)

        # Run CUDA function
        cuda_output = run_cuda_function(cuda_func, inputs, pytorch_output.shape)

        # Compare outputs
        passed, max_diff = compare_outputs(pytorch_output, cuda_output)
        total_diff += max_diff

        if passed:
            passed_tests += 1
            print(f"Test {i+1}: Passed (Max difference: {max_diff:.6f})")
        else:
            print(f"Test {i+1}: Failed (Max difference: {max_diff:.6f})")

    print(f"\nPassed {passed_tests} out of {num_tests} tests")
    print(f"Average max difference: {total_diff / num_tests:.6f}")

if __name__ == "__main__":
    input_file = "example.py"  # Replace with the actual input file path
    judge_transpilation(input_file)
