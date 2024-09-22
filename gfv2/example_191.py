
import torch
import torch.nn.functional as F

class MyModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super(MyModule, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = F.softshrink(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

def fused_dropout_dot_softshrink(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, dropout_p: float) -> torch.Tensor:
    """
    Performs a fused operation of dropout, dot product, and softshrink.
    """
    input_tensor = input_tensor.to(torch.float32)
    weight = weight.to(torch.float32)
    bias = bias.to(torch.float32)

    output = F.linear(input_tensor, weight, bias)
    output = F.dropout(output, p=dropout_p, training=True)
    output = F.softshrink(output)
    return output

function_signature = {
    "name": "fused_dropout_dot_softshrink",
    "inputs": [
        ((1, 10), torch.float32),
        ((10, 5), torch.float32),
        ((5,), torch.float32),
        (0.5, torch.float32)
    ],
    "outputs": [
        ((1, 5), torch.float32),
    ]
}
