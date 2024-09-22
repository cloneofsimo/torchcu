
import torch

def torch_hadamard_any_fp16_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor) -> torch.Tensor:
    """
    Performs Hadamard product (element-wise multiplication) of two tensors, casts to fp16,
    and then returns the result of `torch.any` on the resulting tensor.
    """
    output = torch.mul(input_tensor1, input_tensor2)
    output_fp16 = output.to(torch.float16)
    return torch.any(output_fp16)

function_signature = {
    "name": "torch_hadamard_any_fp16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((), torch.bool),
    ]
}
