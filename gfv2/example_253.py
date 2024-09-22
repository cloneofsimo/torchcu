
import torch
import torch.nn.functional as F

def fused_dropout_cumsum_elementwise_product(input_tensor: torch.Tensor, weight: torch.Tensor, dropout_p: float) -> torch.Tensor:
    """
    Perform a series of operations:
        1. Fused dropout with the given probability.
        2. Element-wise multiplication with the weight tensor.
        3. Cumulative sum along the last dimension.
    """
    input_tensor_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)

    output = F.dropout(input_tensor_int8, p=dropout_p, training=True)
    output = output.mul(weight_int8)
    output = torch.cumsum(output, dim=-1).to(torch.float32)
    return output

function_signature = {
    "name": "fused_dropout_cumsum_elementwise_product",
    "inputs": [
        ((1, 10), torch.float32),
        ((10,), torch.float32),
        (0.5, torch.float32)
    ],
    "outputs": [
        ((1, 10), torch.float32)
    ]
}

