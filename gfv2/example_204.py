
import torch

def exp_cumsum_matrix_exp_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on an input tensor:
    1. Applies exponential function (exp) element-wise
    2. Calculates cumulative sum along the first dimension
    3. Applies matrix exponential (expm) to the result
    """
    input_fp16 = input_tensor.to(torch.float16)
    output = torch.exp(input_fp16)
    output = torch.cumsum(output, dim=0)
    output = torch.linalg.matrix_exp(output)
    return output.to(torch.float32)

function_signature = {
    "name": "exp_cumsum_matrix_exp_function",
    "inputs": [
        ((1, 4, 4), torch.float32),
    ],
    "outputs": [
        ((1, 4, 4), torch.float32),
    ]
}
