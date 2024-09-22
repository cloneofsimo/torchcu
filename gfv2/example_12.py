
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def torch_int8_batchnorm_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                                    running_mean: torch.Tensor, running_var: torch.Tensor, 
                                    eps: float, momentum: float, training: bool,
                                    output_scale: float) -> torch.Tensor:
    """
    Performs int8 batch normalization with optional training and gradient precision scaling.

    Args:
        input_tensor: Input tensor, dtype torch.int8.
        weight: Weight tensor, dtype torch.float32.
        bias: Bias tensor, dtype torch.float32.
        running_mean: Running mean tensor, dtype torch.float32.
        running_var: Running variance tensor, dtype torch.float32.
        eps: Small value added to variance to avoid division by zero.
        momentum: Momentum for running mean and variance update.
        training: Flag indicating if the operation is in training mode.
        output_scale: Output scale factor to apply after batch normalization.

    Returns:
        Output tensor, dtype torch.float32.
    """

    # Convert input to fp32 for batch norm operations
    input_fp32 = input_tensor.to(torch.float32)

    # Batch normalization
    with autocast():
        output = F.batch_norm(input_fp32, weight, bias, running_mean, running_var, 
                              training=training, momentum=momentum, eps=eps)

    # Scale output
    output = output * output_scale

    # Convert back to int8 (optional)
    # output_int8 = output.to(torch.int8) 
    # return output_int8

    return output

function_signature = {
    "name": "torch_int8_batchnorm_function",
    "inputs": [
        ((16, 16, 16, 16), torch.int8),
        ((16,), torch.float32),
        ((16,), torch.float32),
        ((16,), torch.float32),
        ((16,), torch.float32),
        (0.001, torch.float32),
        (0.1, torch.float32),
        (True, torch.bool),
        (1.0, torch.float32)
    ],
    "outputs": [
        ((16, 16, 16, 16), torch.float32),
    ]
}

