
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def audio_downsample_det_backward(input_tensor: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Downsamples an audio signal by a factor, calculates the determinant of the 
    downsampling matrix, and applies the backward operation.
    """
    with autocast():
        downsampled = F.avg_pool1d(input_tensor, kernel_size=factor, stride=factor)
        det_matrix = torch.eye(factor, dtype=torch.float32)
        det = torch.det(det_matrix)

        # Backward operation (upsampling)
        upsampled = F.interpolate(downsampled, scale_factor=factor, mode='linear', align_corners=False)

        return upsampled * det

function_signature = {
    "name": "audio_downsample_det_backward",
    "inputs": [
        ((1, 1024), torch.float32),
        ((), torch.int32),
    ],
    "outputs": [
        ((1, 1024), torch.float32),
    ]
}
