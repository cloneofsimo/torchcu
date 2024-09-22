
import torch
import torch.nn.functional as F

def torch_audio_processing_function(input_tensor: torch.Tensor, filter_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs audio processing steps including:
    - 3D convolution with filter
    - Pixel shuffle for upsampling
    - Zero-crossing rate calculation

    Uses auto-mixed precision (AMP) for efficiency.
    """
    with torch.cuda.amp.autocast():
        # 3D Convolution with filter
        output = F.conv3d(input_tensor, filter_tensor, padding="same")

        # Pixel shuffle for upsampling
        output = torch.nn.functional.pixel_shuffle(output, upscale_factor=2)

        # Calculate zero-crossing rate
        zero_crossings = torch.sum(torch.abs(torch.diff(output, dim=2)) > 0.01, dim=2)
        
        # Normalize ZCR and convert to float32
        output = zero_crossings / (output.shape[2] - 1) 
        output = output.to(torch.float32)

    return output

function_signature = {
    "name": "torch_audio_processing_function",
    "inputs": [
        ((1, 16, 16, 16), torch.float32),  # Input audio (batch, channels, time, freq, height)
        ((1, 16, 3, 3, 3), torch.float32)  # Convolution filter
    ],
    "outputs": [
        ((1, 16, 32, 32, 32), torch.float32),  # Output ZCR values
    ]
}
