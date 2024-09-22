
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class MyModule(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x

def torch_fft_checkpoint_function(input_tensor: torch.Tensor, weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Applies FFT, linear transformation, checkpointing, and fading-in with exponential function.
    """
    # FFT with CUDNN
    x = torch.fft.fft(input_tensor, dim=-1)

    # Apply linear transformation
    x = MyModule(input_tensor.shape[-1], weight.shape[-1])(x)

    # Checkpoint for memory saving
    x = checkpoint(lambda x: MyModule(input_tensor.shape[-1], weight.shape[-1])(x), x)

    # Fading-in with exponential function
    x = (1 - torch.exp(-alpha * torch.arange(x.shape[-1], device=x.device))) * x

    # Inverse FFT
    x = torch.fft.ifft(x, dim=-1)
    return x.real

function_signature = {
    "name": "torch_fft_checkpoint_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((1,), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
