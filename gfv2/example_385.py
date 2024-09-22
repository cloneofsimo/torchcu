
import torch
import torch.nn as nn
import math

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.pe = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        # Replicate positional encoding across batch dimension
        pe = self.pe[:x.size(1), :].unsqueeze(0).repeat(x.size(0), 1, 1)
        return x + pe

class MyFunction(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = LearnedPositionalEncoding(d_model, max_len)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, input_tensor):
        # Replication pad for input
        input_tensor = torch.nn.functional.replication_pad1d(input_tensor, (1, 1))
        # Apply positional encoding
        input_tensor = self.positional_encoding(input_tensor)
        # Linear transformations and bfloat16 casting for speed
        output = self.linear1(input_tensor.to(torch.bfloat16))
        output = torch.nn.functional.relu(output)
        output = self.linear2(output.to(torch.bfloat16))
        output = torch.nn.functional.tanh(output)
        # Cast back to float32 and return
        return output.to(torch.float32)

# Function signature for transpiling
function_signature = {
    "name": "my_function",
    "inputs": [
        ((10, 512), torch.float32),
    ],
    "outputs": [
        ((10, 512), torch.float32),
    ]
}
