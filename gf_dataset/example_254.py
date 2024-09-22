
import torch

def torch_audio_processing(input_tensor: torch.Tensor, 
                           time_stretch_factor: float,
                           filter_kernel: torch.Tensor) -> torch.Tensor:
    """
    Performs audio processing steps including time stretching, convolution, and non-linearity.
    """
    # Time stretching
    stretched_tensor = torch.nn.functional.interpolate(input_tensor.unsqueeze(1),
                                                      scale_factor=time_stretch_factor,
                                                      mode='linear').squeeze(1)

    # Convolution
    convolved_tensor = torch.nn.functional.conv1d(stretched_tensor.unsqueeze(1), filter_kernel.unsqueeze(0))
    convolved_tensor = convolved_tensor.squeeze(1)

    # Softplus activation
    output_tensor = torch.nn.functional.softplus(convolved_tensor.to(torch.float16))
    return output_tensor.to(torch.float32)


function_signature = {
    "name": "torch_audio_processing",
    "inputs": [
        ((16, 1024), torch.float32),  # Input audio (batch, time)
        ((), torch.float32),          # Time stretch factor
        ((128,), torch.float32)       # Filter kernel
    ],
    "outputs": [
        ((16, 1024), torch.float32)  # Processed audio
    ]
}
