
import torch

def gated_linear_units_manhattan_distance(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Applies a gated linear unit (GLU) to the input, followed by pairwise manhattan distance calculation with a weight tensor. 
    """
    input_tensor = input_tensor.to(torch.float32)
    weight = weight.to(torch.float32)
    bias = bias.to(torch.float32)
    
    # GLU activation
    linear_output = torch.nn.functional.linear(input_tensor, weight, bias)
    gate = torch.sigmoid(linear_output[:, ::2])
    output = linear_output[:, 1::2] * gate
    
    # Pairwise Manhattan distance calculation
    distance = torch.cdist(output, weight, p=1)  
    
    return distance.to(torch.int8)

function_signature = {
    "name": "gated_linear_units_manhattan_distance",
    "inputs": [
        ((2, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((2, 4), torch.int8)
    ]
}
