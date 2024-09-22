
import torch

def torch_scatter_add_fp16_function(input_tensor: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs a scatter add operation on a tensor in fp16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    output_fp16 = torch.zeros_like(input_fp16).to(torch.float16)
    output_fp16 = torch.scatter_add(output_fp16, dim, indices, input_fp16)
    return output_fp16.to(torch.float32)

function_signature = {
    "name": "torch_scatter_add_fp16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4,), torch.int32),
        (0, torch.int32),
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
