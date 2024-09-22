
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def torch_sparse_conv_bf16_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Sparse convolution with bfloat16 precision and optional bias.
    Uses unstructured sparsity for the weight tensor.
    """

    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16) if bias is not None else None

    # Pad the input tensor to handle padding for sparse conv
    input_padded = F.pad(input_bf16, (padding, padding, padding, padding), "constant", 0)

    output = F.conv2d(input_padded, weight_bf16, bias_bf16, padding=0)
    output_float = output.to(torch.float32)

    return output_float

function_signature = {
    "name": "torch_sparse_conv_bfloat16_function",
    "inputs": [
        ((1, 1, 4, 4), torch.float32),
        ((1, 1, 3, 3), torch.float32), 
        ((1, 1), torch.float32),
        (0, torch.int32)
    ],
    "outputs": [
        ((1, 1, 2, 2), torch.float32)
    ]
}
