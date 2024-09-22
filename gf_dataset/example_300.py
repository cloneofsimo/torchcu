
import torch
import torch.nn.functional as F

def torch_fading_mixer_bf16_function(input_tensor: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor, 
                                       gamma: float) -> torch.Tensor:
    """
    Applies fading-out and feature mixing using bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight1_bf16 = weight1.to(torch.bfloat16)
    weight2_bf16 = weight2.to(torch.bfloat16)

    # Fading-out: Multiply input by a scalar that decreases with time
    fading_factor = torch.exp(-gamma * input_tensor.shape[1]).to(torch.bfloat16)
    input_bf16 = input_bf16 * fading_factor

    # Feature mixing: Concatenate two linear transformations
    mixed_features = torch.cat(
        [
            torch.matmul(input_bf16, weight1_bf16.t()),
            torch.matmul(input_bf16, weight2_bf16.t())
        ],
        dim=1
    )

    # ReLU activation
    output = F.relu(mixed_features).to(torch.float32)
    return output

function_signature = {
    "name": "torch_fading_mixer_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        (torch.float32,)  # scalar, not tensor
    ],
    "outputs": [
        ((4, 8), torch.float32),
    ]
}
