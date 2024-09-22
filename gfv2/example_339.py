
import torch

def regularized_roll_int8_fp16(input_tensor: torch.Tensor, shift: int, weight: float) -> torch.Tensor:
    """
    Performs a roll operation on the input tensor, applies a regularization term, and converts the result to FP16.
    """
    input_int8 = input_tensor.to(torch.int8)
    rolled_int8 = torch.roll(input_int8, shifts=shift, dims=1)
    regularized_int8 = rolled_int8 - (rolled_int8.mean() * weight)
    return regularized_int8.to(torch.float16)


function_signature = {
    "name": "regularized_roll_int8_fp16",
    "inputs": [
        ((10, 10), torch.float32),
        (1, torch.int32),
        (1, torch.float32)
    ],
    "outputs": [
        ((10, 10), torch.float16),
    ]
}
