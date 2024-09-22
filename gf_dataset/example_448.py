
import torch
from torch import nn

class MyTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        # Multi-head attention
        attn_output, attn_weights = self.multihead_attn(x, x, x, key_padding_mask=attention_mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)

        # Feedforward network
        linear_output = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = self.norm2(x + linear_output)
        return x

def torch_transformer_layer(input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs one transformer layer using MultiHeadAttention and FeedForward network
    """
    layer = MyTransformerLayer(d_model=128, nhead=8)
    output = layer(input_tensor, attention_mask)
    return output

function_signature = {
    "name": "torch_transformer_layer",
    "inputs": [
        ((128, 64), torch.float32),
        ((64, 64), torch.bool)
    ],
    "outputs": [
        ((128, 64), torch.float32)
    ]
}

