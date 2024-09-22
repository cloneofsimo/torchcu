
import torch

def fused_softmax_layer_scaling_decay(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scaling_factor: float, decay_factor: float) -> torch.Tensor:
    """
    Performs a linear transformation followed by fused softmax with layer scaling and decay.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    output = torch.matmul(input_bf16, weight_bf16.t()) + bias_bf16
    output = torch.nn.functional.softmax(output, dim=-1)
    output = output * scaling_factor
    output = output * decay_factor
    return output.to(torch.float32)

function_signature = {
    "name": "fused_softmax_layer_scaling_decay",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
        (torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
