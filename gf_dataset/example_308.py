
import torch
import torch.nn.functional as F

def torch_audio_embedding_function(audio_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Embeds an audio signal using a series of transformations.
    """
    # 1. Normalize audio data
    audio_tensor = audio_tensor.float() / 32768.0  # Assuming audio is in signed 16-bit format
    
    # 2. Apply spectral rolloff
    audio_tensor = torch.stft(audio_tensor, n_fft=1024, hop_length=512)
    spectral_rolloff = torch.mean(audio_tensor, dim=-1, keepdim=True)
    audio_tensor = audio_tensor * spectral_rolloff
    
    # 3. Average pooling
    audio_tensor = F.avg_pool1d(audio_tensor, kernel_size=8, stride=4)
    
    # 4. Permute for matrix multiplication
    audio_tensor = audio_tensor.permute(0, 2, 1)  
    
    # 5. Linear transformation
    output_tensor = torch.matmul(audio_tensor, weights)
    
    # 6. Apply ReLU activation
    output_tensor = F.relu(output_tensor)
    
    # 7. Return the output tensor
    return output_tensor

function_signature = {
    "name": "torch_audio_embedding_function",
    "inputs": [
        ((1, 1, 16000), torch.float32),  # Assuming 16kHz audio signal
        ((513, 128), torch.float32)  # Example weights for linear transformation
    ],
    "outputs": [
        ((1, 128), torch.float32),
    ]
}

