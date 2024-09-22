
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.pe = nn.Parameter(torch.randn(max_len, d_model), requires_grad=True)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class MyModule(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = LearnedPositionalEncoding(d_model)

    def forward(self, x):
        x = self.pos_encoding(x)
        x = self.linear1(x)
        x = torch.nn.functional.elu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a linear transformation, applies ELU activation, and adds a learned positional encoding. 
    """
    model = MyModule(input_tensor.size(-1))
    model = model.to(torch.float16)
    x = model(input_tensor.to(torch.float16))
    return x.to(torch.float32)

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1, 10, 128), torch.float32),
        ((128, 128), torch.float32)
    ],
    "outputs": [
        ((1, 10, 128), torch.float32),
    ]
}
