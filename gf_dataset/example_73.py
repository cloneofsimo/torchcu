
import torch
import torch.nn.functional as F

def torch_hadamard_sigmoid_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Perform a hadamard product (element-wise multiplication) and sigmoid activation.
    """
    output = input_tensor * weight
    return F.sigmoid(output)

function_signature = {
    "name": "torch_hadamard_sigmoid_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
