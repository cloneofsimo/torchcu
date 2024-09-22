
import torch
import torch.nn.functional as F

def torch_audio_feature_extraction(input_tensor: torch.Tensor, 
                                     filter_bank: torch.Tensor, 
                                     window: torch.Tensor) -> torch.Tensor:
    """
    Extract audio features using a filter bank and windowing function.

    Args:
        input_tensor: (batch_size, time_steps) Tensor containing the audio signal.
        filter_bank: (num_filters, time_steps) Tensor representing the filter bank.
        window: (time_steps) Tensor representing the window function.

    Returns:
        (batch_size, num_filters) Tensor containing the extracted audio features.
    """
    # Apply windowing to the audio signal
    windowed_signal = input_tensor * window
    
    # Compute spectral rolloff
    spectral_rolloff = torch.fft.rfft(windowed_signal, dim=1)
    spectral_rolloff = torch.abs(spectral_rolloff)**2 
    spectral_rolloff = torch.roll(spectral_rolloff, 1, dims=1) 
    spectral_rolloff = torch.mean(spectral_rolloff, dim=1)

    # Apply filter bank
    filtered_signal = F.avg_pool1d(spectral_rolloff.unsqueeze(1), kernel_size=256)
    filtered_signal = filtered_signal.squeeze(1)
    filtered_signal = filtered_signal.int8()

    # Interpolate to the desired size
    interpolated_signal = F.interpolate(filtered_signal.float().unsqueeze(1), size=filter_bank.size(0), mode='linear')
    interpolated_signal = interpolated_signal.squeeze(1)

    # Multiply with the filter bank
    features = torch.addcmul(torch.zeros_like(interpolated_signal), 1.0, filter_bank, interpolated_signal.int8())

    # Normalize features
    features = F.normalize(features, p=2, dim=1)

    return features.float()

function_signature = {
    "name": "torch_audio_feature_extraction",
    "inputs": [
        ((16, 1024), torch.float32),  # input_tensor
        ((128, 1024), torch.float32),  # filter_bank
        ((1024,), torch.float32)  # window
    ],
    "outputs": [
        ((16, 128), torch.float32)  # features
    ]
}
