
import torch

class MyConvLogSigmoid(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size)
        self.logsigmoid = torch.nn.LogSigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.logsigmoid(x)
        return x

def conv_logsigmoid_bf16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a 1D convolution followed by log sigmoid activation using bfloat16.
    """
    model = MyConvLogSigmoid(input_tensor.shape[1], weight.shape[0], weight.shape[2])
    model.conv.weight = torch.nn.Parameter(weight.to(torch.bfloat16))
    model.conv.bias = torch.nn.Parameter(bias.to(torch.bfloat16))
    output = model(input_tensor.to(torch.bfloat16))
    return output.to(torch.float32)

function_signature = {
    "name": "conv_logsigmoid_bf16",
    "inputs": [
        ((1, 10, 20), torch.float32),
        ((5, 10, 3), torch.float32),
        ((5,), torch.float32),
    ],
    "outputs": [
        ((1, 5, 18), torch.float32),
    ]
}
