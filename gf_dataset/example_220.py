
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import AdaptiveMaxPool3d

def torch_energy_computation_fp16(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Computes the energy of a 3D signal using a convolution with learnable weights.
    """
    with autocast():
        # Apply convolution with weights in fp16
        conv_output = F.conv3d(input_tensor.to(torch.float16), weights.to(torch.float16), padding=1)
        # Apply adaptive max pooling in fp16
        pooled_output = AdaptiveMaxPool3d((1, 1, 1))(conv_output)
        # Output the energy in fp16
        return pooled_output.to(torch.float16)

function_signature = {
    "name": "torch_energy_computation_fp16",
    "inputs": [
        ((16, 128, 64, 64, 64), torch.float32),
        ((128, 3, 3, 3), torch.float32)
    ],
    "outputs": [
        ((16, 1, 1, 1, 1), torch.float16)
    ]
}
