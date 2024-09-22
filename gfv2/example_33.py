
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class DETRTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation="relu"):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation),
            num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation),
            num_decoder_layers
        )

    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask):
        # Encode the source sequence
        encoder_output = self.encoder(src, src_mask)

        # Decode the target sequence using the encoder output
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, memory_mask)
        return decoder_output

def bucketize(tensor, num_buckets):
    """Bucketize a tensor into a specified number of buckets."""
    return torch.bucketize(tensor, torch.linspace(tensor.min(), tensor.max(), num_buckets))

def affine_grid_generator(theta, size):
    """Generates an affine grid for warping images."""
    return F.affine_grid(theta, size)

@torch.jit.script
def _forward_fn(input_tensor, weight, bias, output_tensor):
    # Perform linear transformation (matrix multiplication)
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output = torch.matmul(input_bf16, weight_bf16.t())

    # Apply bias (if provided)
    if bias is not None:
        output += bias

    # Quantize output to int8
    output_int8 = output.to(torch.int8)
    output_tensor.copy_(output_int8)

def int8_linear_bf16_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Performs a linear transformation (matrix multiplication) using bfloat16, applies optional bias, and quantizes the result to int8.
    """
    output_tensor = torch.empty(input_tensor.shape[0], weight.shape[0], dtype=torch.int8)
    _forward_fn(input_tensor, weight, bias, output_tensor)
    return output_tensor


function_signature = {
    "name": "int8_linear_bf16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, ), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.int8),
    ]
}
