
import torch

def torch_multinomial_baddbmm_function(input_tensor: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a multinomial sampling, followed by a batched matrix multiplication, and then adds a bias term. 
    Uses bfloat16 for computation to improve performance.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weights_bf16 = weights.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    # Multinomial sampling
    samples = torch.multinomial(input_bf16, num_samples=1, replacement=False)

    # Batched matrix multiplication
    result = torch.baddbmm(bias_bf16, weights_bf16, samples.long().unsqueeze(-1), beta=1.0)

    # Element-wise minimum
    output = torch.min(result, torch.tensor(10.0, dtype=torch.bfloat16))

    # Return result as float32
    return output.to(torch.float32)

function_signature = {
    "name": "torch_multinomial_baddbmm_function",
    "inputs": [
        ((10, 5), torch.float32),
        ((5, 3), torch.float32),
        ((10, 3), torch.float32)
    ],
    "outputs": [
        ((10, 3), torch.float32)
    ]
}

