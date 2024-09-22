
import torch
import torch.nn.functional as F

def grid_sampler_elu_spectral_rolloff_function(input_tensor: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Performs grid sampling, applies ELU activation, and calculates spectral rolloff.
    """
    # Perform grid sampling
    sampled_tensor = F.grid_sample(input_tensor.to(torch.float16), grid.to(torch.float16), mode='bilinear', padding_mode='border', align_corners=True)

    # Apply ELU activation
    output = F.elu(sampled_tensor)

    # Calculate spectral rolloff
    spectral_rolloff = torch.fft.rfft(output, dim=1)
    spectral_rolloff = torch.abs(spectral_rolloff)
    rolloff = torch.sum(spectral_rolloff, dim=1) / torch.sum(output, dim=1)

    return rolloff.to(torch.float32)


function_signature = {
    "name": "grid_sampler_elu_spectral_rolloff_function",
    "inputs": [
        ((1, 1, 10, 10), torch.float32),
        ((1, 1, 10, 2), torch.float32)
    ],
    "outputs": [
        ((1, 10), torch.float32),
    ]
}
