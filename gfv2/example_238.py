
import torch

def complex_transform(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor:
      1. Instance normalization
      2. SVD decomposition
      3. Hardsigmoid activation
      4. Matrix multiplication with a learned weight matrix
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    # Instance normalization
    normalized_input = torch.nn.functional.instance_norm(input_bf16, eps=1e-5)

    # SVD decomposition
    u, s, v = torch.linalg.svd(normalized_input)
    
    # Hardsigmoid activation
    hardsigmoid_output = torch.nn.functional.hardsigmoid(s)
    
    # Matrix multiplication with a learned weight matrix
    weight = torch.randn(hardsigmoid_output.size(0), hardsigmoid_output.size(0), dtype=torch.bfloat16, device=input_tensor.device)
    output = torch.matmul(hardsigmoid_output, weight)

    return output.to(torch.float32)

function_signature = {
    "name": "complex_transform",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
