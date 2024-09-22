
import torch
import torchaudio

def audio_processing_function(audio_file: str, 
                             sample_rate: int, 
                             normalization_factor: float) -> torch.Tensor:
    """
    Loads audio from a file, performs normalization and exponential scaling.
    """
    waveform, sample_rate_loaded = torchaudio.load(audio_file)
    assert sample_rate_loaded == sample_rate, "Audio file sample rate does not match expected rate"
    
    # Normalize audio (using mean and stddev)
    waveform_normalized = (waveform - waveform.mean()) / waveform.std()
    
    # Apply exponential scaling
    waveform_scaled = torch.exp(waveform_normalized * normalization_factor)
    
    return waveform_scaled.to(torch.float32)


function_signature = {
    "name": "audio_processing_function",
    "inputs": [
        ((), str),
        ((), int),
        ((), float)
    ],
    "outputs": [
        ((1, -1), torch.float32),
    ]
}
