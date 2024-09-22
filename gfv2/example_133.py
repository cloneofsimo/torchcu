
import torch
import torch.nn.functional as F

def harmonic_percussive_separation_int8(audio: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """
    Performs harmonic-percussive separation using a simple iterative algorithm.
    
    Args:
        audio (torch.Tensor): The input audio signal, shape (N, 1, T).
        iterations (int, optional): Number of iterations to perform. Defaults to 10.
    
    Returns:
        torch.Tensor: The separated harmonic component, shape (N, 1, T).
    """
    
    # Convert to int8
    audio_int8 = audio.to(torch.int8)

    # Initialize harmonic and percussive components
    harmonic = torch.zeros_like(audio_int8)
    percussive = torch.zeros_like(audio_int8)

    # Iterate
    for _ in range(iterations):
        # Perform cross-correlation between harmonic and percussive
        cross_correlation = F.conv1d(harmonic, percussive.flip(dims=[2]), padding="same")

        # Update harmonic component
        harmonic = torch.lerp(harmonic, audio_int8 + cross_correlation, 0.5)
        
        # Update percussive component
        percussive = torch.lerp(percussive, audio_int8 - cross_correlation, 0.5)

    # Convert back to float
    harmonic = harmonic.to(torch.float32)

    return harmonic

function_signature = {
    "name": "harmonic_percussive_separation_int8",
    "inputs": [
        ((1, 1, 1024), torch.float32),
    ],
    "outputs": [
        ((1, 1, 1024), torch.float32),
    ]
}
