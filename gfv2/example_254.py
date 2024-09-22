
import torch

def complex_tensor_operations(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of complex tensor operations, demonstrating various PyTorch functions.
    """
    # 1. Convert to bfloat16
    input_bf16 = input_tensor.to(torch.bfloat16)

    # 2. Apply hardsigmoid activation
    hardsigmoid_output = torch.hardsigmoid(input_bf16)

    # 3. Calculate the eigenvalues of the tensor
    eigenvalues = torch.linalg.eigvals(hardsigmoid_output)

    # 4. Convert eigenvalues to fp32
    eigenvalues_fp32 = eigenvalues.to(torch.float32)

    # 5. Apply ELU activation to eigenvalues
    elu_output = torch.elu(eigenvalues_fp32)

    # 6. Create a tensor filled with ones, with the same shape as input_tensor
    ones_tensor = torch.ones_like(input_tensor, dtype=torch.float32)

    # 7. Multiply the ELU output with the ones tensor
    final_output = elu_output * ones_tensor

    return final_output

function_signature = {
    "name": "complex_tensor_operations",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
