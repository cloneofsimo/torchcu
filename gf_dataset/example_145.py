
import torch

def torch_waveform_analysis(input_tensor: torch.Tensor, window_size: int, step_size: int) -> torch.Tensor:
    """
    Analyzes a waveform by extracting features like mean, variance, and maximum values within sliding windows.
    """
    features = []
    for i in range(0, len(input_tensor) - window_size + 1, step_size):
        window = input_tensor[i:i+window_size]
        features.append(torch.tensor([window.mean(), window.var(), window.max()]))
    return torch.stack(features)

function_signature = {
    "name": "torch_waveform_analysis",
    "inputs": [
        ((100,), torch.float32),
        ((), torch.int32),
        ((), torch.int32)
    ],
    "outputs": [
        ((33,), torch.float32)
    ]
}
