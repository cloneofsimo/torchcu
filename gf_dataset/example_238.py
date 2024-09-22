
import torch
import torch.nn.functional as F

def torch_audio_processing_fp16(input_tensor: torch.Tensor, window_size: int, hop_size: int, 
                                   n_fft: int, mel_bins: int, sample_rate: int, 
                                   mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Performs audio processing steps including:
    1. STFT (Short-Time Fourier Transform)
    2. Mel-spectrogram calculation
    3. Logarithmic scaling
    4. Mean-variance normalization
    
    This function is optimized for speed using FP16 precision and cuDNN.
    """
    input_fp16 = input_tensor.to(torch.float16)
    
    # STFT using cuDNN
    stft = torch.stft(input_fp16, n_fft=n_fft, hop_length=hop_size, win_length=window_size, 
                      center=True, return_complex=True, normalized=False)
    
    # Calculate magnitude spectrogram
    mag_spec = torch.abs(stft)
    
    # Convert to mel-spectrogram
    mel_spec = torch.nn.functional.mel_spectrogram(mag_spec, sample_rate=sample_rate, 
                                                   n_fft=n_fft, hop_length=hop_size, 
                                                   win_length=window_size, 
                                                   center=True, pad_mode="reflect",
                                                   power=2.0, norm="ortho",
                                                   mel_scale="htk",
                                                   n_mels=mel_bins)

    # Logarithmic scaling
    log_mel_spec = torch.log(mel_spec + 1e-6)
    
    # Normalize with provided mean and std
    normalized_mel_spec = (log_mel_spec - mean.to(torch.float16)) / std.to(torch.float16)
    
    # Return normalized mel-spectrogram in FP32
    return normalized_mel_spec.to(torch.float32)

function_signature = {
    "name": "torch_audio_processing_fp16",
    "inputs": [
        ((1, 16000), torch.float32),
        (1, torch.int32),
        (1, torch.int32),
        (1, torch.int32),
        (1, torch.int32),
        (1, torch.int32),
        ((1, 128), torch.float32),
        ((1, 128), torch.float32)
    ],
    "outputs": [
        ((1, 128, 129), torch.float32),
    ]
}

