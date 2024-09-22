
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple

def torch_coord_conv_function(input_tensor: torch.Tensor, filter_size: int) -> torch.Tensor:
    """
    Performs a Coordinate Convolution operation with adaptive max pooling.
    """
    # Calculate coordinate grid
    B, C, H, W = input_tensor.size()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32))
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
    coordinates = torch.stack([grid_x, grid_y], dim=1).to(input_tensor.device)

    # Concatenate coordinates to input tensor
    input_with_coords = torch.cat([input_tensor, coordinates], dim=1)

    # Constant padding
    padding = _triple(filter_size // 2)
    input_with_coords = F.pad(input_with_coords, padding, 'constant', 0)

    # Adaptive max pooling
    output = F.adaptive_max_pool3d(input_with_coords, (filter_size, filter_size))

    # Remove coordinate channels
    output = output[:, :-2, :, :]

    return output

function_signature = {
    "name": "torch_coord_conv_function",
    "inputs": [
        ((1, 3, 32, 32), torch.float32),
        (3, torch.int32)
    ],
    "outputs": [
        ((1, 3, 16, 16), torch.float32),
    ]
}
