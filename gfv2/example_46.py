
import torch
import torch.nn as nn

class FeatureMixingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.decoder_layers = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout), num_decoder_layers)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.decoder_layers(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        x = self.norm(self.linear(x))
        return x

def mixed_transformer_decoder_with_pooling(input_tensor: torch.Tensor, memory_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs feature mixing, adaptive pooling, and transformer decoding.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len).
        memory_tensor (torch.Tensor): Memory tensor of shape (batch_size, d_model, seq_len).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, d_model, seq_len).
    """

    # Feature Mixing
    mixing_block = FeatureMixingBlock(input_tensor.size(1), 128)
    mixed_input = mixing_block(input_tensor.to(torch.int8))
    mixed_input = mixed_input.to(torch.float32)

    # Adaptive Pooling
    pooled_input = nn.AdaptiveAvgPool1d(1)(mixed_input)
    pooled_input = pooled_input.squeeze(-1)

    # Transformer Decoder
    decoder = TransformerDecoder(128, 4, 2)
    output = decoder(pooled_input, memory_tensor)

    return output

function_signature = {
    "name": "mixed_transformer_decoder_with_pooling",
    "inputs": [
        ((10, 100, 20), torch.float32),
        ((10, 128, 20), torch.float32)
    ],
    "outputs": [
        ((10, 128, 20), torch.float32)
    ]
}
