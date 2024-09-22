
import torch
import torch.nn as nn
from torch.nn.functional import adaptive_max_pool3d

class MyModule(nn.Module):
    def __init__(self, padding: int):
        super().__init__()
        self.padding = padding

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        padded_input = torch.nn.functional.pad(input_tensor, (self.padding, self.padding, self.padding, self.padding, self.padding, self.padding), "constant", 0)
        pooled_output = adaptive_max_pool3d(padded_input, (1, 1, 1))
        return pooled_output

function_signature = {
    "name": "my_module_forward",
    "inputs": [
        ((1, 3, 16, 16, 16), torch.float32)
    ],
    "outputs": [
        ((1, 3, 1, 1, 1), torch.float32)
    ]
}
