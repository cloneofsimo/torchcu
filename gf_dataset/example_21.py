
import torch

def torch_rank_logsigmoid_bf16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the matrix rank, applies logsigmoid and returns the result in bfloat16.
    """
    rank = torch.matrix_rank(input_tensor.to(torch.bfloat16))
    logsigmoid_result = torch.logsigmoid(rank)
    return logsigmoid_result.to(torch.bfloat16)

function_signature = {
    "name": "torch_rank_logsigmoid_bf16_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((1,), torch.bfloat16)
    ]
}
