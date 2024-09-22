
import torch
import torch.nn as nn

class AdaptiveAvgPool3DInt8(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool3DInt8, self).__init__()
        self.output_size = output_size

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies adaptive average pooling to the input tensor, using int8 precision.
        """
        input_int8 = input_tensor.to(torch.int8)
        output = torch.nn.functional.adaptive_avg_pool3d(input_int8, self.output_size)
        return output.to(torch.float32)


function_signature = {
    "name": "adaptive_avg_pool3d_int8",
    "inputs": [
        ((16, 3, 32, 32, 32), torch.float32)
    ],
    "outputs": [
        ((16, 3, 4, 4, 4), torch.float32)
    ]
}

