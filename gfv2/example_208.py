
import torch

def example_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on input tensors, including:
    - Einsum with transpose
    - Binary cross-entropy with logits
    - Median calculation
    - Conversion to bfloat16 and int8
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)

    # Einsum with transpose
    output = torch.einsum('ijk,kl->ijl', input_bf16, weight_bf16.T)

    # Binary cross-entropy with logits
    output = torch.sigmoid(output)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, input_tensor)

    # Median calculation
    median = torch.median(output)

    # Conversion to bfloat16 and int8
    output_bf16 = output.to(torch.bfloat16)
    output_int8 = output.to(torch.int8)

    return output_bf16

function_signature = {
    "name": "example_function",
    "inputs": [
        ((4, 4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4, 4), torch.bfloat16),
    ]
}
