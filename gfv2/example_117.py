
import torch
import torch.nn.functional as F

def audio_feature_extraction(input_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Extracts audio features from a raw audio signal.

    Args:
        input_tensor: A 1D tensor representing the raw audio signal.
        sample_rate: The sampling rate of the audio signal.

    Returns:
        A 1D tensor containing the extracted audio features.
    """
    # Interpolate to a fixed length
    target_length = 16000
    interpolated_tensor = F.interpolate(input_tensor.unsqueeze(0), size=target_length, mode='linear').squeeze(0)

    # Calculate the zero-crossing rate (ZCR)
    zcr = torch.sum(torch.abs(torch.diff(interpolated_tensor)), dim=0) / (target_length - 1)

    # Concatenate features
    features = torch.cat([interpolated_tensor, zcr.unsqueeze(0)])

    return features.float()

function_signature = {
    "name": "audio_feature_extraction",
    "inputs": [
        ((16000,), torch.float32),
        (int, None)
    ],
    "outputs": [
        ((16001,), torch.float32)
    ]
}
