
import torch
import torch.nn.functional as F

def conv3d_log_softmax_roll_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Performs a 3D convolution, applies log softmax with temperature scaling, and rolls the output along the last dimension. 
    """
    # Convolution
    output = F.conv3d(input_tensor.to(torch.bfloat16), weight.to(torch.bfloat16), bias.to(torch.bfloat16))
    output = output.to(torch.float32)

    # Log Softmax with temperature scaling
    output = F.log_softmax(output / temperature, dim=1)

    # Roll along the last dimension
    output = torch.roll(output, shifts=1, dims=-1) 

    return output

function_signature = {
    "name": "conv3d_log_softmax_roll_function",
    "inputs": [
        ((1, 16, 32, 32, 32), torch.float32),
        ((16, 16, 3, 3, 3), torch.float32),
        ((16,), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((1, 16, 32, 32, 32), torch.float32)
    ]
}
