
import torch

def torch_elementwise_sum_cumsum_dilation_fp16(input_tensor: torch.Tensor, kernel_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise sum, cumulative sum, and morphological dilation using fp16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    kernel_fp16 = kernel_tensor.to(torch.float16)

    # Element-wise sum
    elementwise_sum = input_fp16 + kernel_fp16

    # Cumulative sum along the last dimension
    cumsum = torch.cumsum(elementwise_sum, dim=-1)

    # Morphological dilation
    dilation = torch.nn.functional.max_pool2d(cumsum.unsqueeze(1), kernel_size=kernel_fp16.shape, stride=1, padding=kernel_fp16.shape // 2)
    dilation = dilation.squeeze(1)

    # Return the dilated result
    return dilation.to(torch.float32)

function_signature = {
    "name": "torch_elementwise_sum_cumsum_dilation_fp16",
    "inputs": [
        ((3, 3, 5, 5), torch.float32),  # Input tensor
        ((3, 3), torch.float32)  # Kernel tensor
    ],
    "outputs": [
        ((3, 3, 5, 5), torch.float32)  # Dilated output
    ]
}
