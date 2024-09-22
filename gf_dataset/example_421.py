
import torch

def torch_bmm_unsqueeze_int8_function(input1: torch.Tensor, input2: torch.Tensor, input3: torch.Tensor) -> torch.Tensor:
    """
    Performs a batch matrix multiplication (bmm), unsqueezes the result, and casts to int8.
    """
    input1_int8 = input1.to(torch.int8)
    input2_int8 = input2.to(torch.int8)
    result = torch.bmm(input1_int8, input2_int8)
    result_unsqueeze = result.unsqueeze(dim=1)
    result_int8 = result_unsqueeze.to(torch.int8)
    return result_int8

function_signature = {
    "name": "torch_bmm_unsqueeze_int8_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((2, 4, 5), torch.float32),
        ((2, 3, 5), torch.float32)
    ],
    "outputs": [
        ((2, 1, 3, 5), torch.int8),
    ]
}
