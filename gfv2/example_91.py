
import torch

def max_filter_adaptive_pool_fp16(input_tensor: torch.Tensor, kernel_size: int, output_size: int) -> torch.Tensor:
    """
    Applies a max filter with the given kernel size, followed by adaptive max pooling to the specified output size.
    The input and output are in fp16 precision.
    """
    input_fp16 = input_tensor.to(torch.float16)
    output = torch.nn.functional.max_pool1d(input_fp16, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    output = torch.nn.functional.adaptive_max_pool1d(output, output_size=output_size)
    return output.to(torch.float32)

function_signature = {
    "name": "max_filter_adaptive_pool_fp16",
    "inputs": [
        ((10, 10), torch.float32),  # Shape (batch_size, input_size)
        (1, torch.int32),           # Kernel size
        (1, torch.int32),           # Output size
    ],
    "outputs": [
        ((10, 1), torch.float32), # Shape (batch_size, output_size)
    ]
}
