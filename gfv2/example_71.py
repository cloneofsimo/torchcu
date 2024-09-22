
import torch

def cross_fade_int8_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Cross-fades between two input tensors using a specified alpha value.

    This function first decomposes the input tensors into their individual components using
    tensor_decomposition. Then, it cross-fades between these components using a weighted average
    with the provided alpha value. Finally, it recombines the cross-faded components into a single
    tensor and returns the result.

    Args:
        input_tensor1 (torch.Tensor): The first input tensor.
        input_tensor2 (torch.Tensor): The second input tensor.
        alpha (float): The cross-fade weight, ranging from 0.0 to 1.0.

    Returns:
        torch.Tensor: The cross-faded tensor.
    """

    # Tensor decomposition for both input tensors
    component1_1, component1_2 = torch.tensor_split(input_tensor1, 2, dim=1)
    component2_1, component2_2 = torch.tensor_split(input_tensor2, 2, dim=1)

    # Cross-fade between components using a weighted average
    cross_faded_component_1 = (1 - alpha) * component1_1 + alpha * component2_1
    cross_faded_component_2 = (1 - alpha) * component1_2 + alpha * component2_2

    # Recombine the cross-faded components
    cross_faded_tensor = torch.cat([cross_faded_component_1, cross_faded_component_2], dim=1)

    # Apply a threshold using a non-equality operation and then apply addcdiv operation
    cross_faded_tensor = torch.where(cross_faded_tensor != 0.0, cross_faded_tensor, 0.0)
    cross_faded_tensor.addcdiv_(cross_faded_tensor, cross_faded_tensor, value=1.0)

    # Convert to int8 for efficient processing
    cross_faded_tensor = cross_faded_tensor.to(torch.int8)

    return cross_faded_tensor

function_signature = {
    "name": "cross_fade_int8_function",
    "inputs": [
        ((2, 2), torch.float32),
        ((2, 2), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((2, 2), torch.int8)
    ]
}
