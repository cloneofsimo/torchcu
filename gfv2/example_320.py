
import torch

def flatten_einsum_fp16_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Flatten input, perform a batched einsum summation with weight, and return the result in fp16.
    """
    input_tensor_fp16 = input_tensor.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)
    flattened_input = input_tensor_fp16.flatten(start_dim=1)
    output = torch.einsum('ij,jk->ik', flattened_input, weight_fp16)
    return output

function_signature = {
    "name": "flatten_einsum_fp16_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((24, 5), torch.float32)
    ],
    "outputs": [
        ((2, 5), torch.float16),
    ]
}
