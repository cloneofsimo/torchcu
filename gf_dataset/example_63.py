
import torch
import torch.nn.functional as F

def torch_image_jacobian_distance_transform(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Jacobian of the input tensor and applies a distance transform 
    to the result, then compares the output with the target tensor.

    Args:
        input_tensor: The input image tensor (e.g., a B x C x H x W tensor).
        target_tensor: The target tensor for comparison (same dimensions as input_tensor).

    Returns:
        A tensor representing the element-wise difference between the transformed input and target.
    """
    # Calculate the Jacobian of the input tensor
    jacobian = torch.autograd.functional.jacobian(lambda x: x, input_tensor)

    # Apply a distance transform to the Jacobian
    transformed_jacobian = F.distance_transform(jacobian, sampling_type="ball", norm_type="l1", eps=1e-06)

    # Calculate the element-wise difference with the target tensor
    output = torch.abs(transformed_jacobian - target_tensor)

    return output

function_signature = {
    "name": "torch_image_jacobian_distance_transform",
    "inputs": [
        ((1, 3, 256, 256), torch.float32),
        ((1, 3, 256, 256), torch.float32)
    ],
    "outputs": [
        ((1, 3, 256, 256), torch.float32),
    ]
}
