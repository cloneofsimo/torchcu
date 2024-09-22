
import torch

def einsum_inner_product_int8(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs an einsum-based inner product with broadcasting and returns an int8 tensor.
    """
    input_tensor_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    output = torch.einsum('ijk,kl->ijl', input_tensor_int8, weight_int8)
    return output.to(torch.int8)


function_signature = {
    "name": "einsum_inner_product_int8",
    "inputs": [
        ((1, 2, 3), torch.float32),
        ((3, 4), torch.float32)
    ],
    "outputs": [
        ((1, 2, 4), torch.int8)
    ]
}
