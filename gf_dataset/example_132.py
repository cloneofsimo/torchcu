
import torch

def torch_elementwise_diff_inplace(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise difference between two tensors in-place on the first input tensor.
    """
    input_tensor1.sub_(input_tensor2)
    return input_tensor1

function_signature = {
    "name": "torch_elementwise_diff_inplace",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
