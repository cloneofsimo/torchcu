
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.cpp_extension import load

# Load the compiled CUDA extension
torch_istft_prune_multinomial_cuda = load(
    name='torch_istft_prune_multinomial_cuda',
    sources=['func.cpp', 'func.cu'],
    extra_cflags=['-O3'],  # Add optimization flags
    extra_cuda_cflags=['-O3']  # Add optimization flags for CUDA
)

# Function signature for the CUDA kernel
function_signature = {
    "name": "torch_istft_prune_multinomial_cuda",
    "inputs": [
        ((1, 1024, 257), torch.float32),  # Input spectrogram
        ((257, 1024), torch.float32),  # Pruning mask
        ((257, 1024), torch.float32),  # Multinomial probabilities
        (1024, torch.int64),  # Hop length
        (256, torch.int64),  # Window length
        (1024, torch.int64),  # Output length
    ],
    "outputs": [
        ((1, 1024, 2), torch.float32),  # Reconstructed audio
    ]
}

def torch_istft_prune_multinomial(spectrogram: torch.Tensor, pruning_mask: torch.Tensor,
                                  multinomial_probs: torch.Tensor, hop_length: int,
                                  window_length: int, output_length: int) -> torch.Tensor:
    """
    Performs an inverse short-time Fourier transform (ISTFT) on the spectrogram,
    applying pruning based on the provided mask and performing multinomial sampling
    based on the multinomial probabilities.
    
    Args:
        spectrogram: Input spectrogram tensor of shape (batch, time, freq).
        pruning_mask: Pruning mask tensor of shape (freq, time), indicating which
                      frequency bins should be pruned.
        multinomial_probs: Multinomial probabilities tensor of shape (freq, time),
                          representing the probability of selecting each frequency bin.
        hop_length: Hop length for ISTFT.
        window_length: Window length for ISTFT.
        output_length: Desired output length for the reconstructed audio.
    
    Returns:
        Reconstructed audio tensor of shape (batch, output_length, 2), where the last
        dimension represents stereo channels.
    """

    # Convert to bfloat16 for faster processing
    spectrogram_bf16 = spectrogram.to(torch.bfloat16)
    pruning_mask_bf16 = pruning_mask.to(torch.bfloat16)
    multinomial_probs_bf16 = multinomial_probs.to(torch.bfloat16)

    # Apply pruning and multinomial sampling
    pruned_spectrogram = spectrogram_bf16 * pruning_mask_bf16
    sampled_spectrogram = pruned_spectrogram * multinomial_probs_bf16

    # Perform ISTFT using CUDA extension
    reconstructed_audio = torch_istft_prune_multinomial_cuda(
        sampled_spectrogram.float(), pruning_mask.float(),
        multinomial_probs.float(), hop_length, window_length, output_length
    )

    return reconstructed_audio.float()
