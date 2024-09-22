
import torch

def interpolate_mean_bf16(input_tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Interpolates the input tensor using linear interpolation and calculates the mean along the last dimension.
    All operations are performed in bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output_bf16 = torch.nn.functional.interpolate(input_bf16, scale_factor=scale, mode='linear', align_corners=False)
    output_mean = torch.mean(output_bf16, dim=-1)
    return output_mean.to(torch.float32)

function_signature = {
    "name": "interpolate_mean_bf16",
    "inputs": [
        ((16, 32, 64), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((16, 32), torch.float32),
    ]
}
