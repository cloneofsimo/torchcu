
import torch

def torch_addcmul_function(input1: torch.Tensor, input2: torch.Tensor, weight: torch.Tensor, value: float) -> torch.Tensor:
    """
    Computes the element-wise product of two tensors, multiplies the product by a scalar weight, and adds it to a third tensor.
    """
    output = torch.addcmul(input1, input2, weight, value=value)
    return output

function_signature = {
    "name": "torch_addcmul_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
