
import torch

def torch_spectral_rolloff_function(input_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Computes the spectral rolloff of an audio signal.

    Args:
        input_tensor: A 2D tensor representing the audio signal, with shape (batch_size, num_frames)
        threshold: The percentage of energy to consider for the spectral rolloff calculation (e.g., 0.85)

    Returns:
        A 1D tensor containing the spectral rolloff values for each audio frame.
    """
    # Calculate the magnitude spectrum of the input audio signal
    magnitude_spectrum = torch.abs(torch.fft.fft(input_tensor, dim=1))

    # Calculate the cumulative sum of the energy spectrum
    cumulative_energy = torch.cumsum(magnitude_spectrum ** 2, dim=1)

    # Calculate the total energy of each frame
    total_energy = cumulative_energy[:, -1]

    # Find the index where the cumulative energy exceeds the threshold
    rolloff_indices = (cumulative_energy >= total_energy * threshold).argmax(dim=1)

    # Calculate the spectral rolloff frequencies
    spectral_rolloff = rolloff_indices.float() / input_tensor.size(1)

    return spectral_rolloff

function_signature = {
    "name": "torch_spectral_rolloff_function",
    "inputs": [
        ((16, 1024), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((16,), torch.float32),
    ]
}
