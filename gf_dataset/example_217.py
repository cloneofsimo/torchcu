
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

class AttentionConvIFFT(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Computes attention, convolution, inverse FFT, and normalization.
        """
        ctx.save_for_backward(query, key, value, weight, bias)
        
        # Attention
        attn = torch.matmul(query, key.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        
        # Convolution
        out = torch.matmul(attn, value)
        
        # Conv-IFFT
        out = F.conv1d(out.transpose(-2, -1), weight, bias).transpose(-2, -1)
        out = torch.fft.ifft(out, dim=-1)
        
        # Norm
        out = F.normalize(out, dim=-1)

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Computes gradients for attention, convolution, inverse FFT, and normalization.
        """
        query, key, value, weight, bias = ctx.saved_tensors
        
        # Backprop through normalization
        grad_output = F.normalize(grad_output, dim=-1)
        
        # Backprop through IFFT
        grad_output = torch.fft.fft(grad_output, dim=-1)
        
        # Backprop through convolution
        grad_output = F.conv1d(grad_output.transpose(-2, -1), weight, bias).transpose(-2, -1)

        # Backprop through attention
        grad_value = torch.matmul(grad_output.transpose(-2, -1), attn)
        grad_attn = torch.matmul(grad_output, value.transpose(-2, -1))

        # Backprop through key/query matmul
        grad_key = torch.matmul(grad_attn.transpose(-2, -1), query)
        grad_query = torch.matmul(grad_attn, key)

        # Backprop through convolution weights/bias
        grad_weight = grad_output.transpose(-2, -1)
        grad_bias = grad_output.sum(dim=-2)

        return grad_query, grad_key, grad_value, grad_weight, grad_bias

# Function signature
function_signature = {
    "name": "attention_conv_ifft",
    "inputs": [
        ((16, 128, 64), torch.float32),
        ((16, 128, 64), torch.float32),
        ((16, 128, 64), torch.float32),
        ((3, 64, 64), torch.float32),
        ((64,), torch.float32),
    ],
    "outputs": [
        ((16, 128, 64), torch.float32),
    ]
}
