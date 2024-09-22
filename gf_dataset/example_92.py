
import torch

def torch_signal_shift_function(input_tensor: torch.Tensor, shift_amount: int) -> torch.Tensor:
    """
    Shifts a signal (represented by a tensor) by a specified amount.
    """
    return torch.roll(input_tensor, shifts=shift_amount, dims=1)

function_signature = {
    "name": "torch_signal_shift_function",
    "inputs": [
        ((10, 100), torch.float32),
        ((), torch.int32)
    ],
    "outputs": [
        ((10, 100), torch.float32),
    ]
}
