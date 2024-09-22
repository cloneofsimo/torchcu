
import torch
import torch.nn.functional as F

def torch_audio_normalization_grouped_conv(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, norm_mean: torch.Tensor, norm_std: torch.Tensor) -> torch.Tensor:
    """
    Performs audio normalization, grouped convolution, and ReLU activation.
    """
    # Normalize audio
    normalized_input = (input_tensor - norm_mean) / norm_std
    
    # Grouped convolution
    output = F.conv1d(normalized_input, weight, bias, groups=4)

    # ReLU activation
    output = F.relu(output)
    
    return output

function_signature = {
    "name": "torch_audio_normalization_grouped_conv",
    "inputs": [
        ((1, 160, 1024), torch.float32),
        ((160, 160, 3), torch.float32),
        ((160,), torch.float32),
        ((160,), torch.float32),
        ((160,), torch.float32),
    ],
    "outputs": [
        ((1, 160, 1021), torch.float32)
    ]
}
