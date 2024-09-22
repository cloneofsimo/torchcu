
import torch
import torch.nn.functional as F

def torch_function(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor,  gamma: float) -> torch.Tensor:
    """
    Performs a series of operations on tensors:
    1. Hadamard product of x and y, with both tensors converted to bfloat16.
    2. Applies a gradient penalty to the result.
    3. Performs a pitch correction using a convolution with weight.
    4. Applies ReLU activation.
    5. Returns the result as a float32 tensor.

    Args:
        x: First input tensor.
        y: Second input tensor.
        weight: Convolution kernel for pitch correction.
        gamma: Gradient penalty coefficient.

    Returns:
        The output tensor as a float32 tensor.
    """
    # Convert to bfloat16 for Hadamard product
    x_bf16 = x.to(torch.bfloat16)
    y_bf16 = y.to(torch.bfloat16)
    
    # Hadamard product
    hadamard_product = x_bf16 * y_bf16

    # Gradient penalty
    grad_penalty = torch.mean((torch.linalg.norm(torch.autograd.grad(hadamard_product.sum(), hadamard_product, create_graph=True)[0], dim=1) - 1.0) ** 2)
    hadamard_product = hadamard_product + gamma * grad_penalty

    # Pitch correction (Convolution with weight)
    hadamard_product = hadamard_product.to(torch.float32) # back to float32 for convolution
    output = F.conv1d(hadamard_product.unsqueeze(1), weight, padding="same")

    # ReLU activation
    output = F.relu(output)

    # Return as float32
    return output.squeeze(1).to(torch.float32)


function_signature = {
    "name": "torch_function",
    "inputs": [
        ((4, 8), torch.float32),
        ((4, 8), torch.float32),
        ((1, 3, 3), torch.float32),
        (torch.float32,)
    ],
    "outputs": [
        ((4, 8), torch.float32),
    ]
}
