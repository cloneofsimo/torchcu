
import torch
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd

@custom_fwd(cast_inputs=torch.bfloat16)
def fused_gelu_rfft_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Fused GELU activation, complex-valued FFT, and matrix multiplication.
    """
    # GELU activation (fused with bfloat16)
    gelu = x * torch.sigmoid(1.702 * x)

    # Complex-valued FFT
    rfft = torch.fft.rfft(gelu, dim=-1)

    # Matrix multiplication (using bf16)
    output = torch.matmul(rfft, weight.to(torch.bfloat16).t())

    return output.to(torch.float32)


@custom_bwd
def fused_gelu_rfft_bf16_backward(ctx, grad_output):
    """
    Backward pass for fused_gelu_rfft_bf16.
    """
    x, weight = ctx.inputs
    rfft = torch.fft.rfft(x.to(torch.bfloat16) * torch.sigmoid(1.702 * x.to(torch.bfloat16)), dim=-1)

    # Gradient computation
    grad_weight = torch.matmul(grad_output.to(torch.bfloat16), rfft.conj().t())
    grad_x = torch.fft.irfft(torch.matmul(grad_output.to(torch.bfloat16), weight.to(torch.bfloat16)), dim=-1) * \
             (1.702 * x.to(torch.bfloat16) * torch.sigmoid(1.702 * x.to(torch.bfloat16)) * (1 - torch.sigmoid(1.702 * x.to(torch.bfloat16))) + \
             torch.sigmoid(1.702 * x.to(torch.bfloat16)))

    return grad_x.to(torch.float32), grad_weight.to(torch.float32)


function_signature = {
    "name": "fused_gelu_rfft_bf16",
    "inputs": [
        ((8, 16, 32), torch.float32),
        ((16, 64), torch.float32),
    ],
    "outputs": [
        ((8, 16, 64), torch.float32),
    ]
}

