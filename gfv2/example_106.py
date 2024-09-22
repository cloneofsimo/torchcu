
import torch

def laplace_filter_fp16(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Applies a Laplace filter to an input tensor using fp16 precision.
    """
    input_fp16 = input_tensor.to(torch.float16)
    kernel_fp16 = kernel.to(torch.float16)
    output = torch.nn.functional.conv2d(input_fp16, kernel_fp16, padding=1)
    return torch.true_divide(output, torch.sum(kernel_fp16)).to(torch.float32)

function_signature = {
    "name": "laplace_filter_fp16",
    "inputs": [
        ((2, 1, 4, 4), torch.float32),
        ((1, 1, 3, 3), torch.float32)
    ],
    "outputs": [
        ((2, 1, 4, 4), torch.float32),
    ]
}
