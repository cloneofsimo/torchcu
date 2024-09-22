
import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

def transformer_layer_bf16(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a single transformer layer operation using bfloat16 precision.
    """
    layer = TransformerLayer(d_model=512, nhead=8).to(torch.bfloat16)
    output = layer(input_tensor.to(torch.bfloat16))
    return output.to(torch.float32)

function_signature = {
    "name": "transformer_layer_bf16",
    "inputs": [
        ((1, 10, 512), torch.float32)
    ],
    "outputs": [
        ((1, 10, 512), torch.float32)
    ]
}
