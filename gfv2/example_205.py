
import torch

def addcmul_function(input_tensor: torch.Tensor, tensor1: torch.Tensor, tensor2: torch.Tensor, value: float) -> torch.Tensor:
    """
    Performs a simple addcmul operation on input tensor with tensors 1 and 2, and a given value.
    """
    output = input_tensor.addcmul(tensor1, tensor2, value=value)
    return output

function_signature = {
    "name": "addcmul_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
