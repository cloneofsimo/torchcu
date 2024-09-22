
import torch
import torch.nn.functional as F

def torch_conv1d_softmax_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Applies a 1D convolution, then performs log softmax along the last dimension, 
    and finally applies a cumulative sum.
    """
    # Convert to fp16 for potential performance gains on some hardware
    input_tensor = input_tensor.to(torch.float16)
    weight = weight.to(torch.float16)
    bias = bias.to(torch.float16)

    # Apply 1D convolution
    output = F.conv1d(input_tensor, weight, bias)

    # Log softmax along last dimension
    output = F.log_softmax(output, dim=-1)

    # In-place cumulative sum
    output.cumsum(dim=-1, out=output)

    # Return to fp32
    return output.to(torch.float32)

function_signature = {
    "name": "torch_conv1d_softmax_function",
    "inputs": [
        ((10, 3, 20), torch.float32), 
        ((3, 2, 5), torch.float32), 
        ((2), torch.float32)
    ],
    "outputs": [
        ((10, 3, 20), torch.float32),
    ]
}
