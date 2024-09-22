
import torch
import torch.nn.functional as F
from cutlass import *

def torch_regularized_spectral_rolloff_function(input_tensor: torch.Tensor, weight: torch.Tensor, 
                                                 bias: torch.Tensor, lambda_reg: float) -> torch.Tensor:
    """
    Applies a linear transformation, regularizes with spectral roll-off, and calculates the distance transform. 
    """
    
    # Linear transformation
    output = F.linear(input_tensor, weight, bias)
    
    # Spectral Roll-off Regularization
    spectral_rolloff = torch.sum(torch.square(torch.fft.fft(weight, dim=1)))
    output = output - lambda_reg * spectral_rolloff
    
    # Distance Transform (using PyTorch's implementation)
    output = F.distance_transform(output.unsqueeze(1), sampling_type='point', norm='L2', return_indices=False)
    
    # Layer Scaling
    output = output * 0.5
    
    return output.to(torch.float32)


function_signature = {
    "name": "torch_regularized_spectral_rolloff_function",
    "inputs": [
        ((16, 16), torch.float32),
        ((16, 16), torch.float32),
        ((16,), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((16, 16), torch.float32),
    ]
}

