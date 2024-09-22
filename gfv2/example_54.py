
import torch

def threshold_bfloat16_function(input_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Applies a threshold to the input tensor using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    threshold_bf16 = torch.tensor(threshold, dtype=torch.bfloat16)
    output_bf16 = torch.where(input_bf16 > threshold_bf16, input_bf16, torch.tensor(0.0, dtype=torch.bfloat16))
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "threshold_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
