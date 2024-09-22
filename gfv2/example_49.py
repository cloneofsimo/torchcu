
import torch
import torch.nn as nn

def pruned_softshrink_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor, prune_threshold: float) -> torch.Tensor:
    """
    Performs model pruning, soft shrink, and quantization to int8 for a simple linear layer.
    """
    # 1. Pruning: Apply a threshold to the weight tensor
    pruned_weight = torch.where(torch.abs(weight) > prune_threshold, weight, torch.zeros_like(weight))

    # 2. Soft Shrink: Apply soft shrink to the weight tensor
    softshrink_weight = nn.functional.softshrink(pruned_weight, lambd=0.5)

    # 3. Quantization to int8: Quantize the weights and inputs to int8
    weight_int8 = softshrink_weight.to(torch.int8)
    input_int8 = input_tensor.to(torch.int8)

    # 4. Matrix Multiplication: Perform matrix multiplication in int8
    output_int8 = torch.matmul(input_int8, weight_int8.t())

    # 5. Dequantization: Convert the output back to float32
    output = output_int8.to(torch.float32)
    return output

function_signature = {
    "name": "pruned_softshrink_int8_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
