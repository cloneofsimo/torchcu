
import torch
import torch.nn.functional as F

def my_fp16_function(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor, including:
        - Adaptive average pooling
        - Einsum outer product with weights
        - ReLU activation
        - In-place FP16 conversion
    """
    input_tensor = input_tensor.to(torch.float16)
    weights = weights.to(torch.float16)
    pooled = F.adaptive_avg_pool1d(input_tensor, 1)
    output = torch.einsum('b,ij->bij', pooled.squeeze(2), weights)
    output = F.relu(output)
    output.to(torch.float16, copy=False)  # In-place conversion to FP16
    return output

function_signature = {
    "name": "my_fp16_function",
    "inputs": [
        ((3, 2, 5), torch.float32),
        ((2, 5), torch.float32),
    ],
    "outputs": [
        ((3, 2, 5), torch.float16),
    ]
}
