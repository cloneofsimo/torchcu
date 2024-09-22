import torch
import torch.nn.functional as F

def torch_conv2d_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Perform a 2D convolution and ReLU activation.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)
    
    output = F.conv2d(input_bf16, weight_bf16, bias_bf16)
    
    return torch.relu(output).to(torch.float32)

function_signature = [
    'torch_conv2d_function',
    ('input_tensor', (1, 3, 32, 32), torch.float32),  # (batch_size, in_channels, height, width)
    ('weight', (6, 3, 5, 5), torch.float32),         # (out_channels, in_channels, kernel_height, kernel_width)
    ('bias', (6,), torch.float32)                    # (out_channels)
]