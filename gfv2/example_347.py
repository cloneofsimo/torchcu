
import torch

def instance_norm_inplace(input_tensor: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Performs instance normalization in-place on the input tensor.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-5.

    Returns:
        torch.Tensor: The input tensor with instance normalization applied in-place.
    """
    mean = input_tensor.mean(dim=(2, 3), keepdim=True)
    std = input_tensor.std(dim=(2, 3), keepdim=True)
    input_tensor.sub_(mean).div_(std + eps)
    return input_tensor

function_signature = {
    "name": "instance_norm_inplace",
    "inputs": [
        ((2, 3, 4, 5), torch.float32),
        (float)
    ],
    "outputs": [
        ((2, 3, 4, 5), torch.float32),
    ]
}
