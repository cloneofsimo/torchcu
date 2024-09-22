
import torch
import torch.nn as nn

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = (x / keep_prob) * random_tensor
        return output

def torch_drop_path_sqrt_function(input_tensor: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """
    Applies drop path with sqrt scaling and returns the square root of the input tensor.
    """
    output = DropPath(drop_prob)(input_tensor)
    return torch.sqrt(output)

function_signature = {
    "name": "torch_drop_path_sqrt_function",
    "inputs": [
        ((2, 2, 2), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((2, 2, 2), torch.float32)
    ]
}
