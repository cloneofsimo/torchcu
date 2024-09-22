
import torch
import torch.nn.functional as F

def torch_pitch_shift_scharr_gradient_bf16(input_tensor: torch.Tensor, pitch_shift: int, scharr_kernel: torch.Tensor) -> torch.Tensor:
    """
    Applies pitch shift and Scharr gradient to an input tensor.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    shifted = torch.nn.functional.phase_vocoder(input_bf16, torch.tensor(pitch_shift / 100.0))
    scharr_x = F.conv2d(shifted, scharr_kernel, padding=1)
    scharr_y = F.conv2d(shifted, scharr_kernel.t(), padding=1)
    gradient = torch.sqrt(torch.square(scharr_x) + torch.square(scharr_y))
    return gradient.to(torch.float32)


function_signature = {
    "name": "torch_pitch_shift_scharr_gradient_bf16",
    "inputs": [
        ((1, 1, 256, 256), torch.float32),
        (1, torch.int32),
        ((3, 3), torch.float32)
    ],
    "outputs": [
        ((1, 1, 256, 256), torch.float32)
    ]
}
