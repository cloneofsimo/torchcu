
import torch

def torch_conv_bf16_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int) -> torch.Tensor:
    """
    Perform a 1D convolution with bfloat16 precision, including padding and stride.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    # Unfold the input tensor
    unfolded_input = torch.nn.functional.unfold(input_bf16, kernel_size=weight.shape[2], stride=stride, padding=padding)

    # Matrix multiplication with the unfolded input
    output_bf16 = torch.matmul(unfolded_input, weight_bf16.reshape(weight.shape[0], -1).t())

    # Add bias
    output_bf16 += bias_bf16.reshape(1, -1)

    # Apply ReLU activation
    output_bf16 = torch.relu(output_bf16)

    # Reshape the output to the correct dimensions
    output_bf16 = output_bf16.reshape(input_tensor.shape[0], weight.shape[0], -1)

    # Convert back to float32
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "torch_conv_bf16_function",
    "inputs": [
        ((1, 1, 10), torch.float32),
        ((1, 1, 3), torch.float32),
        ((1,), torch.float32),
        (1, ),
        (1, ),
    ],
    "outputs": [
        ((1, 1, 8), torch.float32),
    ]
}
