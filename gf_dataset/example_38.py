
import torch
import torch.nn.functional as F

def torch_grid_sample_selu_int8(input_tensor: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Performs grid sampling on the input tensor, applies SELU activation, and converts to int8.
    """
    output = F.grid_sample(input_tensor.to(torch.float32), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    output = F.selu(output)
    return output.to(torch.int8)

function_signature = {
    "name": "torch_grid_sample_selu_int8",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((1, 1, 224, 224), torch.float32)
    ],
    "outputs": [
        ((1, 3, 224, 224), torch.int8),
    ]
}
