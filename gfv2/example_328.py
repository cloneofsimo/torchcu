
import torch

def expand_var_bmm_bf16_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs the following operations:
    1. Expands the input tensor to match the weight's dimensions.
    2. Calculates the variance of the expanded input tensor along the last dimension.
    3. Performs a batched matrix multiplication with the weight tensor in bfloat16 precision.
    4. Applies a bias tensor.
    5. Quantizes the result to int8.
    """

    # Expand the input tensor
    expanded_input = input_tensor.expand(weight.shape[0], *input_tensor.shape[1:])

    # Calculate variance
    variance = expanded_input.var(dim=-1, keepdim=True)

    # Convert to bfloat16
    expanded_input_bf16 = expanded_input.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)

    # Batched matrix multiplication
    output_bf16 = torch.bmm(expanded_input_bf16, weight_bf16.transpose(1, 2))

    # Apply bias
    output_bf16 = output_bf16 + bias.to(torch.bfloat16)

    # Quantize to int8
    output_int8 = output_bf16.to(torch.int8)

    return output_int8, variance

function_signature = {
    "name": "expand_var_bmm_bf16_int8_function",
    "inputs": [
        ((1, 2, 3), torch.float32),
        ((4, 3, 5), torch.float32),
        ((4, 5), torch.float32)
    ],
    "outputs": [
        ((4, 2, 5), torch.int8),
        ((4, 1), torch.float32)
    ]
}
