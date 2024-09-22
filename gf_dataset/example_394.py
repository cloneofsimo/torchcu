
import torch
import torch.nn.functional as F

def torch_audio_feature_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates zero-crossing rate and weight sparsity for an audio input tensor.
    """
    # Zero-crossing rate
    zero_crossings = torch.sum(torch.abs(torch.diff(input_tensor, dim=1)) > 0.0, dim=1).float() / (input_tensor.shape[1] - 1)

    # Weight sparsity
    weight_sparsity = torch.sum(weight == 0) / weight.numel()

    return zero_crossings.to(torch.float32), weight_sparsity.to(torch.float32)

function_signature = {
    "name": "torch_audio_feature_function",
    "inputs": [
        ((1, 16000), torch.float32),
        ((10, 10), torch.float32),
    ],
    "outputs": [
        ((1,), torch.float32),
        ((1,), torch.float32)
    ]
}
