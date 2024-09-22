
import torch
import torch.nn.functional as F

def torch_svd_cross_gradient_bf16(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs SVD decomposition on the input tensor, calculates the Roberts cross-gradient,
    and applies a ReLU activation, all using bfloat16 precision. 
    Returns the result as a float32 tensor.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    U, S, V = torch.linalg.svd(input_bf16)
    
    # Roberts cross-gradient calculation
    gradient_x = F.conv2d(input_bf16, torch.tensor([[1, 0], [0, -1]], dtype=torch.bfloat16), padding=1)
    gradient_y = F.conv2d(input_bf16, torch.tensor([[0, 1], [-1, 0]], dtype=torch.bfloat16), padding=1)
    gradient = torch.sqrt(gradient_x**2 + gradient_y**2)

    # Combine SVD and gradient results, apply ReLU
    output = torch.matmul(U, torch.diag(S) @ V.t()) + gradient
    output = torch.relu(output).to(torch.float32)
    return output

function_signature = {
    "name": "torch_svd_cross_gradient_bf16",
    "inputs": [
        ((16, 16), torch.float32)
    ],
    "outputs": [
        ((16, 16), torch.float32)
    ]
}
