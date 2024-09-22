
import torch
import torch.nn.functional as F

def fading_in_function(input_tensor: torch.Tensor, weights: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Applies a fading-in effect to the input tensor. 
    
    Args:
        input_tensor: Input tensor to be faded in (shape: [batch_size, channels, height, width]).
        weights: A tensor of weights for each step (shape: [steps]).
        steps: The total number of steps for fading in.

    Returns:
        A tensor representing the faded-in output (shape: [batch_size, channels, height, width]).
    """
    batch_size, channels, height, width = input_tensor.shape
    output = torch.zeros_like(input_tensor)

    for i in range(steps):
        output += input_tensor * weights[i]

    return output

function_signature = {
    "name": "fading_in_function",
    "inputs": [
        ((4, 3, 224, 224), torch.float32), 
        ((10,), torch.float32),
        (10,) 
    ],
    "outputs": [
        ((4, 3, 224, 224), torch.float32),
    ]
}
