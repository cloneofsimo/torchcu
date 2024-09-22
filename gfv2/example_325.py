
import torch
import torch.nn.functional as F

def depthwise_conv2d_topk_bf16(input_tensor: torch.Tensor, weight: torch.Tensor, threshold: float) -> list[torch.Tensor]:
    """
    Performs a depthwise convolution, applies a threshold, and then finds the top-k values.
    Uses bfloat16 for intermediate computations and returns the top-k values and their indices as a list.

    Args:
        input_tensor: Input tensor of shape (batch_size, in_channels, height, width).
        weight: Depthwise convolution weight tensor of shape (in_channels, 1, kernel_size, kernel_size).
        threshold: Threshold value for filtering the output.

    Returns:
        A list containing two tensors:
            - Top-k values: Tensor of shape (batch_size, k).
            - Indices of the top-k values: Tensor of shape (batch_size, k).
    """
    # Convert input and weights to bfloat16
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    
    # Perform depthwise convolution in bfloat16
    output_bf16 = F.conv2d(input_bf16, weight_bf16, groups=input_tensor.shape[1])
    
    # Apply threshold
    output_bf16 = torch.where(output_bf16 > threshold, output_bf16, torch.zeros_like(output_bf16))
    
    # Find the top-k values in each batch
    batch_size = input_tensor.shape[0]
    k = 5  # Fixed k for this example (can be adjusted)
    topk_values, topk_indices = torch.topk(output_bf16.view(batch_size, -1), k=k, dim=1)

    # Convert back to float32
    topk_values = topk_values.to(torch.float32)
    topk_indices = topk_indices.to(torch.float32)
    
    return [topk_values, topk_indices]

function_signature = {
    "name": "depthwise_conv2d_topk_bf16",
    "inputs": [
        ((1, 4, 10, 10), torch.float32),
        ((4, 1, 3, 3), torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((1, 5), torch.float32),
        ((1, 5), torch.float32)
    ]
}
