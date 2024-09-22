
import torch
import torch.nn.functional as F

def my_complex_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scale: float, decay: float) -> torch.Tensor:
    """
    Performs a complex sequence of operations:
        1. Linear transformation
        2. PReLU activation
        3. 3D transposed convolution
        4. Layer scaling and decay
    """
    output = F.linear(input_tensor, weight, bias)
    output = F.prelu(output, weight=torch.tensor([0.25], dtype=torch.float32))  # PReLU with fixed slope
    output = F.conv_transpose3d(output.unsqueeze(1), weight, bias=None, stride=2, padding=1, output_padding=1)
    output = output.squeeze(1)
    output *= scale * torch.exp(-decay * output)  # Layer scaling and decay (inplace operation)
    return output

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((1, 4, 4, 4), torch.float32),
        ((16, 4, 4, 4), torch.float32),
        ((16,), torch.float32),
        (torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((1, 16, 8, 8, 8), torch.float32),
    ]
}
