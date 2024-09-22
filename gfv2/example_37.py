
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def fused_linear_pixel_unshuffle_mixup_eq_fp16_backward(input_tensor: torch.Tensor, 
                                                       weight: torch.Tensor, 
                                                       bias: torch.Tensor, 
                                                       mixup_lambda: float, 
                                                       target_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a fused operation of linear, pixel unshuffle, mixup, equality comparison,
    and backward propagation in fp16 precision.

    Args:
        input_tensor: Input tensor of shape (N, C, H, W).
        weight: Weight tensor for the linear layer of shape (C_out, C_in).
        bias: Bias tensor for the linear layer of shape (C_out).
        mixup_lambda: Mixup interpolation factor (between 0 and 1).
        target_tensor: Target tensor of shape (N, C_out, H_out, W_out).

    Returns:
        A tuple containing:
            - The output tensor after mixup, of shape (N, C_out, H_out, W_out).
            - The gradient of the input tensor, of shape (N, C, H, W).
    """

    with autocast():
        # Linear layer with fp16 weights
        output = F.linear(input_tensor.to(torch.float16), weight.to(torch.float16), bias.to(torch.float16))

        # Pixel unshuffle
        output = F.pixel_unshuffle(output, downscale_factor=2)

        # Mixup
        output_mixed = mixup_lambda * output + (1 - mixup_lambda) * target_tensor.to(torch.float16)

        # Equality comparison (element-wise)
        eq_mask = (output_mixed == target_tensor.to(torch.float16)).to(torch.float16)

        # Backward pass (calculating gradient of input)
        output_mixed.backward(eq_mask)

    return output_mixed.to(torch.float32), input_tensor.grad.to(torch.float32)

function_signature = {
    "name": "fused_linear_pixel_unshuffle_mixup_eq_fp16_backward",
    "inputs": [
        ((16, 32, 32, 32), torch.float32),
        ((64, 32), torch.float32),
        ((64,), torch.float32),
        (torch.float32),
        ((16, 64, 16, 16), torch.float32)
    ],
    "outputs": [
        ((16, 64, 16, 16), torch.float32),
        ((16, 32, 32, 32), torch.float32),
    ]
}

