
import torch
import torch.nn.functional as F

def torch_morphological_dilation_fp16_int8(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Performs morphological dilation with a kernel on an input tensor.
    Input is assumed to be fp16, kernel is int8, and output is int8.
    """
    # Convert to fp16 and flatten
    input_tensor_fp16 = input_tensor.to(torch.float16).flatten()
    kernel_int8 = kernel.to(torch.int8)
    # Perform dilation
    output_int8 = F.conv1d(input_tensor_fp16.unsqueeze(0).unsqueeze(0), kernel_int8.unsqueeze(0), padding=kernel.shape[0] // 2)
    # Reshape and return int8 output
    output_int8 = output_int8.squeeze().reshape(input_tensor.shape)
    return output_int8.to(torch.int8)

function_signature = {
    "name": "torch_morphological_dilation_fp16_int8",
    "inputs": [
        ((1, 1, 1, 1), torch.float16),
        ((1, 1, 1, 1), torch.int8)
    ],
    "outputs": [
        ((1, 1, 1, 1), torch.int8),
    ]
}
