
import torch

def softshrink_crossfade_fp32(input_tensor: torch.Tensor, threshold: float, weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Applies the soft shrink operation to the input tensor, followed by a cross-fade with another tensor.
    The operation is performed in FP32.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        threshold (float): The threshold for the soft shrink operation.
        weight (torch.Tensor): The tensor to cross-fade with.
        alpha (float): The cross-fade weight.

    Returns:
        torch.Tensor: The output tensor.
    """
    output = torch.where(input_tensor.abs() > threshold, input_tensor - threshold * torch.sign(input_tensor), torch.zeros_like(input_tensor))
    output = output * (1 - alpha) + weight * alpha
    return output

function_signature = {
    "name": "softshrink_crossfade_fp32",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32),
        ((4, 4), torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
