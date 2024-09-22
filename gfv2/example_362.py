
import torch
import torch.nn.functional as F

def complex_function(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    A complex function demonstrating various operations, including:
    - Unfold
    - Softplus
    - Softsign
    - Elementwise min
    - Forward (for testing)
    - fp16, bf16, fp32 conversions
    """
    # Convert to fp16 for potential speedup
    input_tensor = input_tensor.to(torch.float16)
    weights = weights.to(torch.float16)

    # Unfold the input tensor
    unfolded = F.unfold(input_tensor, kernel_size=(3, 3), padding=1)

    # Apply softplus activation
    unfolded = F.softplus(unfolded)

    # Perform matrix multiplication with weights
    output = torch.matmul(unfolded, weights.t())

    # Apply softsign activation
    output = torch.sigmoid(output) - 0.5

    # Apply elementwise minimum with a constant
    output = torch.min(output, torch.tensor(0.75, dtype=torch.float16))

    # Convert back to fp32
    output = output.to(torch.float32)

    return output

function_signature = {
    "name": "complex_function",
    "inputs": [
        ((4, 4, 4), torch.float32),
        ((36, 10), torch.float32),
    ],
    "outputs": [
        ((4, 4, 10), torch.float32),
    ]
}
