
import torch

def conv_permute_unfold(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a 2D convolution with permute and unfold operations. 
    """
    # Permute input tensor to NCHW format
    input_tensor = input_tensor.permute(0, 3, 1, 2)

    # Unfold input tensor
    unfolded_input = torch.nn.functional.unfold(input_tensor, kernel_size=weight.shape[2:])

    # Matrix multiplication with weight
    output = unfolded_input @ weight.view(weight.shape[0], -1)

    # Add bias
    output = output + bias.view(1, -1)

    # Reshape and permute output
    output = output.view(input_tensor.shape[0], weight.shape[0], *input_tensor.shape[2:])
    output = output.permute(0, 2, 3, 1)

    return output

function_signature = {
    "name": "conv_permute_unfold",
    "inputs": [
        ((1, 3, 5, 5), torch.float32),
        ((2, 3, 3, 3), torch.float32),
        ((2,), torch.float32)
    ],
    "outputs": [
        ((1, 5, 5, 2), torch.float32),
    ]
}

