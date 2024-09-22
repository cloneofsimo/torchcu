
import torch
from torch.nn.functional import conv2d

def torch_spectrogram_int8_sub(input_tensor: torch.Tensor, window: torch.Tensor, stride: int, n_fft: int, 
                                 filter_tensor: torch.Tensor, bias: torch.Tensor, 
                                 mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Calculates the spectrogram of an input tensor, subtracts a filter, and normalizes. 
    All operations are performed in int8 precision for efficiency.
    """

    # Convert to int8
    input_int8 = input_tensor.to(torch.int8)
    window_int8 = window.to(torch.int8)
    filter_int8 = filter_tensor.to(torch.int8)
    bias_int8 = bias.to(torch.int8)

    # Calculate spectrogram
    spectrogram = torch.stft(input_int8, n_fft=n_fft, hop_length=stride, window=window_int8)
    spectrogram_abs = torch.abs(spectrogram)
    
    # Subtract filter
    spectrogram_sub = spectrogram_abs - filter_int8

    # Gather relevant frequency bins
    spectrogram_sub = spectrogram_sub[:, :, :spectrogram_sub.size(2) // 2]

    # Convert back to float32 and normalize
    spectrogram_sub = spectrogram_sub.to(torch.float32)
    spectrogram_sub = (spectrogram_sub - mean) / std

    return spectrogram_sub

function_signature = {
    "name": "torch_spectrogram_int8_sub",
    "inputs": [
        ((1, 1, 16000), torch.float32),
        ((1, 1, 512), torch.float32),
        (1, torch.int32),
        (512, torch.int32),
        ((1, 1, 257, 129), torch.float32),
        ((1, 1, 257, 129), torch.float32),
        ((1, 1, 257, 129), torch.float32),
        ((1, 1, 257, 129), torch.float32)
    ],
    "outputs": [
        ((1, 1, 257, 129), torch.float32),
    ]
}
