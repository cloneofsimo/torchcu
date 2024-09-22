
import torch

def noise_injection_median_pad(input_tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Applies noise injection, calculates median, and pads the input tensor.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    noise = torch.randn_like(input_bf16) * noise_level
    noisy_input = input_bf16 + noise.to(torch.bfloat16)
    median = torch.median(noisy_input, dim=1, keepdim=True).values
    padded_tensor = torch.nn.functional.pad(median, (1, 1, 1, 1), 'constant', 0.0)
    return padded_tensor.to(torch.float32)

function_signature = {
    "name": "noise_injection_median_pad",
    "inputs": [
        ((1, 10), torch.float32),
        (torch.float32, )  # Noise level (scalar)
    ],
    "outputs": [
        ((1, 12), torch.float32)
    ]
}
