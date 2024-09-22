from pathlib import Path
import runpy
import subprocess
import numpy as np
import torch
import ctypes
from typing import Callable

from rich.table import Table
import traceback
import multiprocessing

table = Table(title="Transpilation Results")
table.add_column("File", style="cyan", no_wrap=True)
table.add_column("Score", style="magenta")


CURR_DIR = Path(__file__).parent.parent
DATA_DIR = CURR_DIR / "test_data"

def load_pytorch_function(file_path: str, function_name: str) -> Callable:
    """Load the PyTorch function from the given file."""
    module = runpy.run_path(file_path)
    return module[function_name]

def load_function_signature(file_path: str) -> tuple[str, list[tuple[tuple, torch.dtype], list[tuple[tuple, torch.dtype]]]]:
    """Load the function signature from the given file."""
    module = runpy.run_path(file_path)
    signature_dict = module['function_signature']
    name = signature_dict['name']
    inputs = signature_dict['inputs']
    outputs = signature_dict['outputs']

    # validate the args follow the format
    # ((shape,), dtype)
    for arg in (inputs + outputs):
        if not isinstance(arg, tuple):
            raise ValueError(f"Invalid argument: {arg}")

        if len(arg) != 2:
            raise ValueError(f"Invalid argument: {arg}")

        if not isinstance(arg[0], tuple):
            raise ValueError(f"Invalid argument: {arg}")
        if len(arg[0]) < 1:
            raise ValueError(f"Invalid argument: {arg}")
        if not all(isinstance(dim, int) for dim in arg[0]):
            raise ValueError(f"Invalid argument: {arg}")

        if not isinstance(arg[1], torch.dtype):
            raise ValueError(f"Invalid argument: {arg}")

    return name, inputs, outputs


def transpile_to_cuda(file_path: str) -> Path:
    """Transpile the PyTorch function to CUDA code."""
    output_file = Path(file_path).with_suffix('.cu')
    print(f"Transpiling to CUDA: {output_file}")
    return output_file

def compile_cuda(cuda_file: Path) -> Path:
    """Compile the CUDA code to a shared library."""
    output_file = cuda_file.with_suffix('.so')
    subprocess.run(['nvcc', '-Xcompiler', '-fPIC', '--shared', '-o', str(output_file), str(cuda_file), '-lcublas'], check=True)
    return output_file

def prepare_inputs(signature: list) -> list:
    """Prepare input tensors based on the function signature."""
    positional_arguments = []
    for arg in signature:
        positional_arguments.append(torch.randn(arg[0], dtype=arg[1]))

    return positional_arguments

def run_pytorch_function(func: Callable, inputs: list) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Run the PyTorch function with the given inputs."""
    with torch.no_grad():
        return func(*inputs)


def load_cuda_function(lib_path: Path, function_name: str, inputs: list, outputs: list) -> Callable:
    lib_path = lib_path.resolve()
    lib = ctypes.CDLL(lib_path)
    func = getattr(lib, function_name)
    args = [
        ctypes.c_int # num_args
    ]

    # ('input_tensor', (4, 4), torch.float32)
    # ('weight', (4, 4), torch.float32)
    # ('other', (2, 4, 4), torch.float32)
    for arg in inputs:
        arg_shape, arg_dtype = arg

        # pointer to the arg_dtype
        args.append(ctypes.POINTER(ctypes.c_float))
        for dim in arg_shape:
            args.append(ctypes.c_int)

    for output in outputs:
        output_shape, output_dtype = output
        # We do not need to include the shape of the output tensor
        # for dim in output_shape:
        #     args.append(ctypes.c_int)

    # print("Loading CUDA function\n", f"func({', '.join(str(arg) for arg in args)})")
    func.argtypes = args
    func.restype = None
    return func

def run_cuda_function(func: Callable, inputs: list[torch.Tensor], outputs: list[torch.Tensor]):
    for arg in inputs:
        print("Input shape:", arg.shape)
    for arg in outputs:
        print("Output shape:", arg.shape)

    args = []

    for arg in inputs:
        arr = arg.cpu().numpy()
        # Add args as ctypes.POINTER(ctypes.c_float) and then the shape
        args.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        for dim in arr.shape:
            args.append(ctypes.c_int(dim))

    for arg in outputs:
        arr = arg.cpu().numpy()
        args.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    args = [ctypes.c_int(len(args))] + args

    print("Running CUDA function\n", f"func({', '.join(str(arg) for arg in args)})")
    func(*args)

def compare_outputs(pytorch_output: torch.Tensor, cuda_output: torch.Tensor, rtol: float = 1e-2, atol: float = 1e-2) -> bool:
    """Compare the outputs of PyTorch and CUDA implementations."""
    max_diff = np.max(np.abs(pytorch_output.cpu().numpy() - cuda_output.cpu().numpy()))
    allclose = torch.allclose(pytorch_output.cpu(), cuda_output.cpu(), rtol=rtol, atol=atol)
    return allclose, max_diff

def judge_transpilation(input_file: Path, num_tests: int = 10) -> None:
    """Judge the correctness of the transpiled CUDA code."""
    print(f"Testing transpilation of {input_file}")

    # Load PyTorch function and signature
    function_name, inputs_signature, outputs_signature = load_function_signature(input_file)
    pytorch_func = load_pytorch_function(input_file, function_name)

    # print(f"Loaded PyTorch function: {function_name} with signature {inputs_signature}")

    # Transpile and compile CUDA code
    cuda_file = transpile_to_cuda(input_file)
    lib_path = compile_cuda(cuda_file)
    cuda_func = load_cuda_function(lib_path, function_name, inputs_signature, outputs_signature)

    passed_tests = 0
    total_diff = 0

    for i in range(num_tests):
        # Prepare inputs
        inputs_args = prepare_inputs(inputs_signature)

        # Run PyTorch function
        pytorch_output = run_pytorch_function(pytorch_func, inputs_args)
        if not isinstance(pytorch_output, tuple):
            pytorch_output = (pytorch_output,)
        pytorch_output = list(pytorch_output)

        # Run CUDA function
        # Prepare outputs for CUDA function
        cuda_outputs = []
        for output in pytorch_output:
            cuda_outputs.append(torch.empty_like(output))
        run_cuda_function(cuda_func, inputs_args, cuda_outputs)

        # Compare outputs
        test_passed = True
        test_diff = 0
        for i, _ in enumerate(pytorch_output):
            passed, max_diff = compare_outputs(pytorch_output[i], cuda_outputs[i])
            test_passed &= passed
            test_diff += max_diff

        total_diff += test_diff
        passed_tests += passed

    return passed_tests / num_tests

def judge_it(input_file: Path, num_tests: int = 10) -> float:
    """Judge the correctness of the transpiled CUDA code in a separate process."""

    def target(queue):
        try:
            score = judge_transpilation(input_file, num_tests)
            queue.put(score)
        except Exception as e:
            print("Error in judge_it_process", e)
            traceback.print_exc()
            queue.put(0.0)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(queue,))
    process.start()
    process.join()

    return queue.get() if not queue.empty() else 0.0
