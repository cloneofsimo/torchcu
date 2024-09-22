
import torch
import torch.nn.functional as F
import numpy as np

def my_tensor_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, input_tensor3: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on input tensors, demonstrating various PyTorch functionalities.
    """
    # 1. Block Diagonal: Create a block diagonal matrix from input_tensor1
    block_diag_tensor = torch.block_diag(*torch.split(input_tensor1, 1, dim=0))

    # 2. Exponential: Apply exponential function to input_tensor2
    exponential_tensor = torch.exp(input_tensor2)

    # 3. Affine Grid Generation: Create an affine grid using input_tensor3
    grid = F.affine_grid(input_tensor3, (1, 1, 1, 1), align_corners=False)

    # 4. Multiplication and ReLU Activation (using bfloat16):
    block_diag_tensor = block_diag_tensor.to(torch.bfloat16)
    exponential_tensor = exponential_tensor.to(torch.bfloat16)
    grid = grid.to(torch.bfloat16)

    intermediate_tensor = block_diag_tensor * exponential_tensor * grid
    intermediate_tensor = intermediate_tensor.to(torch.float32)
    output_tensor = F.relu(intermediate_tensor)

    # 5. Check for numerical closeness (fp16)
    output_tensor = output_tensor.to(torch.float16)
    if torch.isclose(output_tensor, torch.tensor(0.0, dtype=torch.float16), atol=1e-3):
        print("Output tensor is close to zero.")

    return output_tensor

function_signature = {
    "name": "my_tensor_function",
    "inputs": [
        ((2, 2, 2, 2), torch.float32),
        ((2, 2, 2, 2), torch.float32),
        ((2, 6), torch.float32)
    ],
    "outputs": [
        ((2, 2, 2, 2), torch.float16),
    ]
}
