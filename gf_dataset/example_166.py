
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout=dropout
            ),
            num_encoder_layers
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        return self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

def torch_transformer_encoder_function(input_tensor: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Perform a transformer encoder operation with element-wise power and adaptive average pooling.
    """
    encoder = TransformerEncoder(d_model=8, nhead=2, num_encoder_layers=2, dim_feedforward=16)
    
    output = encoder(input_tensor.to(torch.float32), src_mask=mask.to(torch.bool))
    output = torch.pow(output, 2.0, out=output)  # Element-wise power in-place
    output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
    return output.to(torch.bfloat16)

function_signature = {
    "name": "torch_transformer_encoder_function",
    "inputs": [
        ((16, 8, 4, 4), torch.float32),
        ((16, 8, 4, 4), torch.float32),
        ((16, 4, 4), torch.bool),
    ],
    "outputs": [
        ((16, 8, 1, 1), torch.bfloat16),
    ]
}
