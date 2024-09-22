
import torch

def pool_outer_sum(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Performs 3D max pooling, outer product, and element-wise summation on the input tensor.

    Args:
        input_tensor: Input tensor with shape (B, C, D, H, W)
        weights: Weights tensor with shape (C)

    Returns:
        Output tensor with shape (B, D, H, W)
    """

    # Convert to int8
    input_tensor = input_tensor.to(torch.int8)
    weights = weights.to(torch.int8)

    # 3D Max Pooling
    pooled = torch.nn.functional.max_pool3d(input_tensor, kernel_size=3, stride=2, padding=1)

    # Outer Product
    output = torch.einsum('bchw,c->bhdw', pooled, weights)

    # Elementwise Sum
    output = output + torch.sum(pooled, dim=1, keepdim=True)

    # Convert back to fp16
    output = output.to(torch.float16)
    return output

function_signature = {
    "name": "pool_outer_sum",
    "inputs": [
        ((2, 4, 8, 16, 32), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((2, 8, 8, 16), torch.float16),
    ]
}

