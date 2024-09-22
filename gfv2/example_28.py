
import torch
import torch.nn as nn
import torch.nn.functional as F

class Cutout(nn.Module):
    def __init__(self, size):
        super(Cutout, self).__init__()
        self.size = size

    def forward(self, x):
        n_batch, _, h, w = x.size()
        y = torch.zeros_like(x)
        mask = torch.ones_like(x)
        for batch_idx in range(n_batch):
            # Cutout random region
            x0 = torch.randint(0, h, (1,))
            y0 = torch.randint(0, w, (1,))
            x1 = torch.clamp(x0 + self.size, 0, h)
            y1 = torch.clamp(y0 + self.size, 0, w)
            mask[batch_idx, :, x0:x1, y0:y1] = 0
            y[batch_idx] = x[batch_idx] * mask[batch_idx]
        return y

def cutout_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies cutout to an input tensor.
    """
    cutout = Cutout(size=16)
    return cutout(input_tensor)

function_signature = {
    "name": "cutout_function",
    "inputs": [
        ((1, 3, 32, 32), torch.float32)
    ],
    "outputs": [
        ((1, 3, 32, 32), torch.float32)
    ]
}
