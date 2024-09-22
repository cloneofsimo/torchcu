
import torch
import torch.nn.functional as F

def token_mixing_stochastic_depth_bf16(input_tensor: torch.Tensor, weight: torch.Tensor, 
                                       bias: torch.Tensor, dropout_prob: float) -> torch.Tensor:
    """
    Performs token mixing with stochastic depth using bfloat16 precision.
    """

    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    # Token mixing
    output = F.linear(input_bf16, weight_bf16, bias_bf16)

    # Stochastic depth
    if dropout_prob > 0:
        output = F.dropout(output, p=dropout_prob, training=True)

    # Return to float32
    return output.to(torch.float32)

function_signature = {
    "name": "token_mixing_stochastic_depth_bf16",
    "inputs": [
        ((16, 32, 128), torch.float32),
        ((128, 128), torch.float32),
        ((128,), torch.float32),
        (0.1,)
    ],
    "outputs": [
        ((16, 32, 128), torch.float32)
    ]
}
