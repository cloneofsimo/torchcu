
import torch

def torch_fp32_elementwise_sum_function(input_tensors: list[torch.Tensor], mode: str = "le") -> torch.Tensor:
    """
    Performs element-wise sum of multiple tensors with optional mode specification.

    Args:
        input_tensors: A list of tensors of the same shape.
        mode: 'le' for less-than-equal, 'ge' for greater-than-equal. Defaults to 'le'.

    Returns:
        A tensor representing the element-wise sum of the input tensors.
    """

    if len(input_tensors) == 0:
        raise ValueError("Input list of tensors cannot be empty.")

    # Ensure all tensors have the same shape
    first_shape = input_tensors[0].shape
    for tensor in input_tensors:
        if tensor.shape != first_shape:
            raise ValueError("All tensors must have the same shape.")

    # Initialize result tensor
    result = input_tensors[0].clone()

    # Perform element-wise sum with mode
    if mode == "le":
        for tensor in input_tensors[1:]:
            result = torch.where(tensor <= result, result, tensor)
            result += tensor
    elif mode == "ge":
        for tensor in input_tensors[1:]:
            result = torch.where(tensor >= result, result, tensor)
            result += tensor
    else:
        raise ValueError("Invalid mode. Must be 'le' or 'ge'.")

    return result.squeeze()


function_signature = {
    "name": "torch_fp32_elementwise_sum_function",
    "inputs": [
        ([((4, 4), torch.float32)], "list"),
        (("le", ), torch.int32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
