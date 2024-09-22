
import torch

class MyModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModule, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.avg_pool1d(x.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        x = self.fc2(x)
        return x.to(torch.float32)

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a simple linear transformation with bfloat16, then applies GELU and average pooling.
    """
    model = MyModule(input_tensor.shape[1], 16, 10)
    model.fc1.weight.data = weight.to(torch.bfloat16)
    output = model(input_tensor.to(torch.bfloat16))
    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((10, 4), torch.float32),  # Input tensor
        ((16, 4), torch.float32)  # Weight tensor
    ],
    "outputs": [
        ((10, 10), torch.float32),  # Output tensor
    ]
}
