
import torch

def conv_bf16_diagflat(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a convolution followed by a diagonal flattening, all in bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output_bf16 = torch.conv_tbc(input_bf16, weight_bf16)
    output_bf16 = torch.diagflat(output_bf16.squeeze(1), dim1=1, dim2=2)
    return output_bf16.to(torch.float16)

function_signature = {
    "name": "conv_bf16_diagflat",
    "inputs": [
        ((1, 3, 10, 10), torch.float32),
        ((3, 3, 3, 3), torch.float32)
    ],
    "outputs": [
        ((1, 9, 10, 10), torch.float16),
    ]
}
