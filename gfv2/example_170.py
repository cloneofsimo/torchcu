
import torch

def zero_padding_fp32_function(input_tensor: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Pads the input tensor with zeros on all sides.
    """
    return torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding), 'constant', 0).to(torch.float32)

function_signature = {
    "name": "zero_padding_fp32_function",
    "inputs": [
        ((3, 3), torch.float32),
        (int, int)
    ],
    "outputs": [
        ((3 + 2*2, 3 + 2*2), torch.float32)
    ]
}
