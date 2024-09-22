
import torch
import numpy as np

def torch_int8_conv_preact_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Perform an int8 convolution with pre-activation and bias, returning the output tensor.
    """
    input_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    bias_int8 = bias.to(torch.int8)

    output_int8 = torch.nn.functional.conv2d(input_int8, weight_int8, bias_int8, padding=1)
    output = output_int8.to(torch.float32)  # Convert back to float32

    # Pre-activation (ReLU in this example)
    output = torch.nn.functional.relu(output, inplace=True)

    return output

function_signature = {
    "name": "torch_int8_conv_preact_function",
    "inputs": [
        ((1, 3, 28, 28), torch.float32),
        ((3, 3, 3, 3), torch.float32),
        ((3,), torch.float32)
    ],
    "outputs": [
        ((1, 3, 28, 28), torch.float32),
    ]
}
