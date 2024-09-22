import torch

def linear_transformation(input_tensor: torch.Tensor, weight_tensor: torch.Tensor, bias_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies a linear transformation to the input tensor.

    Args:
    input_tensor (torch.Tensor): The input tensor to be transformed.
    weight_tensor (torch.Tensor): The weight tensor for the linear transformation.
    bias_tensor (torch.Tensor): The bias tensor for the linear transformation.

    Returns:
    torch.Tensor: The transformed tensor.
    """
    # Check if the input tensor and weight tensor have compatible shapes
    if input_tensor.shape[1] != weight_tensor.shape[0]:
        raise ValueError("Input tensor and weight tensor have incompatible shapes")

    # Apply the linear transformation
    output_tensor = torch.matmul(input_tensor, weight_tensor) + bias_tensor

    return output_tensor



# function_signature
function_signature = {
    "name": "linear_transformation",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [((4, 4), torch.float32)]
}