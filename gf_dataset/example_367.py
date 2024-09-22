
import torch
import torch.nn.functional as F

def torch_upsample_sum_trace_function(input_tensor: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    Upsamples the input tensor, sums along the last dimension, traces the resulting matrix, and returns the result.
    """
    upsampled = F.interpolate(input_tensor, scale_factor=scale_factor, mode='nearest')
    summed = upsampled.sum(dim=-1)
    trace = summed.trace()
    return trace.unsqueeze(0)

function_signature = {
    "name": "torch_upsample_sum_trace_function",
    "inputs": [
        ((4, 4, 4), torch.float32),
        (1, torch.int32),
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
