
import torch

def weight_standardized_bf16_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a simple linear transformation (matrix multiplication) with weight standardization and activation using bfloat16.
    """
    # Weight standardization
    weight_mean = weight.mean()
    weight_std = weight.std()
    weight_standardized = (weight - weight_mean) / weight_std

    # Convert to bfloat16
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight_standardized.to(torch.bfloat16)

    # Matrix multiplication
    output = torch.matmul(input_bf16, weight_bf16.t())

    # ReLU activation and return as float32
    return torch.relu(output).to(torch.float32)

function_signature = {
    "name": "weight_standardized_bf16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
