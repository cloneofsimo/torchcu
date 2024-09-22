
import torch

def transposed_conv3d_example(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a 3D transposed convolution with bias, followed by ReLU activation.
    """
    output = torch.nn.functional.conv_transpose3d(input_tensor, weight, bias=bias, stride=2, padding=1, output_padding=1)
    return torch.relu(output)

function_signature = {
    "name": "transposed_conv3d_example",
    "inputs": [
        ((2, 3, 4, 4, 4), torch.float32),  # Input tensor
        ((3, 2, 3, 3, 3), torch.float32),  # Weight tensor
        ((3,), torch.float32),  # Bias tensor
    ],
    "outputs": [
        ((2, 3, 8, 8, 8), torch.float32),  # Output tensor
    ]
}
