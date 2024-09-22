
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout),
            num_encoder_layers
        )

    def forward(self, src):
        return self.encoder(src)

def transformer_encoder_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a transformer encoder operation.
    """
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    encoder = TransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
    output = encoder(input_tensor)
    return output

function_signature = {
    "name": "transformer_encoder_function",
    "inputs": [
        ((1, 100, 512), torch.float32)
    ],
    "outputs": [
        ((1, 100, 512), torch.float32),
    ]
}
