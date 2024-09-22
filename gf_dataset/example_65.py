
import torch

def torch_fold_function(input_tensor: torch.Tensor, axis: int, initial_value: torch.Tensor) -> torch.Tensor:
    """
    Performs a fold operation along a specified axis.
    """
    assert input_tensor.dim() > 1, "Input tensor must have at least 2 dimensions."
    assert 0 <= axis < input_tensor.dim(), "Axis must be within tensor dimensions."
    
    # Reshape to fold along the specified axis
    input_reshaped = input_tensor.transpose(axis, -1).reshape(-1, input_tensor.shape[axis])
    
    # Fold operation
    folded = torch.cumsum(input_reshaped, dim=1) + initial_value.unsqueeze(0)
    
    # Reshape back to original dimensions (except for folded axis)
    output_shape = list(input_tensor.shape)
    output_shape[axis] = 1
    folded = folded.reshape(output_shape)
    return folded

function_signature = {
    "name": "torch_fold_function",
    "inputs": [
        ((16, 4, 4), torch.float32),
        (0, torch.int32),
        ((1,), torch.float32)
    ],
    "outputs": [
        ((16, 1, 4), torch.float32)
    ]
}
