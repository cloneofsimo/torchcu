
import torch

def torch_bmm_out_bfloat16_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a batch matrix multiplication and activation using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output = torch.bmm(input_bf16, weight_bf16)
    return torch.relu(output).to(torch.float32)

function_signature = {
    "name": "torch_bmm_out_bfloat16_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((2, 4, 5), torch.float32)
    ],
    "outputs": [
        ((2, 3, 5), torch.float32),
    ]
}
