
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

def torch_transformer_layer_int8_function(input_tensor: torch.Tensor, src_mask: torch.Tensor, 
                                       src_key_padding_mask: torch.Tensor, 
                                       d_model: int, nhead: int, dim_feedforward: int, 
                                       dropout: float, activation: str = "relu") -> torch.Tensor:
    """
    Performs a single TransformerEncoderLayer with int8 precision and returns the output tensor.
    """
    input_tensor = input_tensor.to(torch.int8)
    layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation=activation)
    output = layer(input_tensor, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    return output.to(torch.float32)


function_signature = {
    "name": "torch_transformer_layer_int8_function",
    "inputs": [
        ((10, 10, 10), torch.float32),
        ((10, 10), torch.bool),
        ((10,), torch.bool),
        (10, torch.int),
        (10, torch.int),
        (10, torch.int),
        (0.1, torch.float32),
        ("relu", torch.str)
    ],
    "outputs": [
        ((10, 10, 10), torch.float32),
    ]
}
