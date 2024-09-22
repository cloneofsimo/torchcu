
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Define the CUDA extension
module = load(
    name="custom_cuda_module",
    sources=["func.cu"],
    extra_include_paths=["/usr/local/cuda/include", "/usr/include/"],
    extra_cflags=["-std=c++11"],
    extra_ldflags=["-lcudart", "-lcublas"],
    verbose=True,
)


def learned_positional_encoding_bf16_max_filter(input_tensor: torch.Tensor, learned_positional_encoding: torch.Tensor) -> torch.Tensor:
    """
    Applies learned positional encoding, converts to bfloat16, performs max filtering, and returns the result in float32.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    learned_positional_encoding_bf16 = learned_positional_encoding.to(torch.bfloat16)

    output_bf16 = input_bf16 + learned_positional_encoding_bf16
    output_bf16 = F.max_pool2d(output_bf16, kernel_size=3, stride=1, padding=1)  # Max filtering

    return output_bf16.to(torch.float32)


function_signature = {
    "name": "learned_positional_encoding_bf16_max_filter",
    "inputs": [
        ((4, 3, 128, 128), torch.float32),
        ((1, 3, 128, 128), torch.float32)
    ],
    "outputs": [
        ((4, 3, 128, 128), torch.float32),
    ]
}
