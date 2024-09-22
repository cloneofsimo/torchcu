
import torch

def torch_isclose_int8_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08) -> torch.Tensor:
    """
    Compares two tensors element-wise for equality within a tolerance.
    Returns a tensor of the same shape as the input tensors, where each element is 1 if the corresponding elements
    in the input tensors are considered close, and 0 otherwise.
    """
    result = torch.isclose(input_tensor1.to(torch.float32), input_tensor2.to(torch.float32), rtol=rtol, atol=atol).to(torch.int8)
    return result

function_signature = {
    "name": "torch_isclose_int8_function",
    "inputs": [
        ((4, 4), torch.int8),
        ((4, 4), torch.int8),
        (None, torch.float32),
        (None, torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.int8),
    ]
}
