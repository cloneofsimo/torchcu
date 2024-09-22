
import torch
import torch.nn as nn

class LayerNormMinLogspace(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNormMinLogspace, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # Convert to bfloat16
        x = x.to(torch.bfloat16)

        # Layer normalization
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + self.eps)

        # Minimum operation
        min_val = torch.min(x, dim=-1, keepdim=True).values

        # Logspace transformation
        x = torch.logspace(min_val, x, base=2.0, dim=-1)

        # Scale and shift
        x = self.gamma * x + self.beta

        # Convert back to float32
        return x.to(torch.float32)

def torch_layer_norm_min_logspace(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies Layer Normalization, Minimum, and Logspace transformation to an input tensor.
    """
    layer_norm = LayerNormMinLogspace(input_tensor.shape[1:])
    output = layer_norm(input_tensor)
    return output

function_signature = {
    "name": "torch_layer_norm_min_logspace",
    "inputs": [
        ((4, 4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4, 4), torch.float32),
    ]
}
