
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, num_layers=1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        src = self.norm(src)
        for layer in self.layers:
            src = layer(src, src)
        return src

def transformer_encoder_diag_backward(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Applies a TransformerEncoder layer with a diagonal weight matrix, then calculates the backward pass.
    """
    d_model = input_tensor.shape[-1]
    nhead = 4
    encoder = TransformerEncoder(d_model, nhead)
    output = encoder(input_tensor)
    output = output.mul(weight.diag())
    output.backward(torch.ones_like(output))
    return input_tensor.grad

function_signature = {
    "name": "transformer_encoder_diag_backward",
    "inputs": [
        ((1, 10, 512), torch.float32),
        ((512,), torch.float32)
    ],
    "outputs": [
        ((1, 10, 512), torch.float32),
    ]
}
