
import torch

def custom_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations, including:
    - Matrix multiplication with `weight`
    - Addition with `bias`
    - Pixel unshuffle
    - ReLU activation
    - Element-wise less than comparison
    - Backwards pass (optional)
    """
    input_tensor = input_tensor.to(torch.float16)
    weight = weight.to(torch.float16)
    bias = bias.to(torch.float16)

    output = torch.addmv(input_tensor, weight, bias)
    output = torch.pixel_unshuffle(output, downscale_factor=2)
    output = torch.relu(output)

    mask = torch.lt(output, 0.5)
    output = output * mask  # In-place modification

    # Backward pass (optional, not implemented)
    # ...

    return output.to(torch.float32)

function_signature = {
    "name": "custom_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((1, 1), torch.float32),
    ]
}
