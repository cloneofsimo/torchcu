
import torch

def torch_audio_feature_function(input_tensor: torch.Tensor, sample_rate: float, window_size: int, hop_length: int) -> torch.Tensor:
    """
    Calculates the spectral centroid of an audio signal, applies rotary positional embedding, 
    and slices the resulting tensor.
    """
    # 1. Calculate the spectral centroid
    spec_centroid = torch.stft(input_tensor, n_fft=window_size, hop_length=hop_length, return_complex=False)
    spec_centroid = torch.mean(spec_centroid * torch.arange(spec_centroid.shape[1], dtype=torch.float32), dim=1)

    # 2. Apply rotary positional embedding
    # (This is a simplified version, assuming fixed embedding)
    positions = torch.arange(spec_centroid.shape[0], dtype=torch.float32)
    frequencies = positions * (2 * torch.pi / sample_rate)
    rotary_embedding = torch.stack([torch.cos(frequencies), torch.sin(frequencies)], dim=-1)
    spec_centroid = spec_centroid * rotary_embedding

    # 3. Slice the tensor
    sliced_output = spec_centroid[::2, :]

    return sliced_output.to(torch.float32)

function_signature = {
    "name": "torch_audio_feature_function",
    "inputs": [
        ((1000,), torch.float32),  # Example input shape
        (16000, ),  # Sample rate
        (1024,),  # Window size
        (512,)  # Hop length
    ],
    "outputs": [
        ((500, 2), torch.float32)
    ]
}

