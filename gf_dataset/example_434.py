
import torch

def torch_fused_softmax_std_hardtanh_bf16(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs fused operations on a tensor:
        1. Softmax with dim=1
        2. Standard deviation along dim=1
        3. Hardtanh with min_val=-2, max_val=2
    All operations are performed in bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    softmax_output = torch.softmax(input_bf16, dim=1)
    std_output = torch.std(softmax_output, dim=1, keepdim=True)
    output = torch.hardtanh(std_output, min_val=-2, max_val=2).to(torch.float32)
    return output

function_signature = {
    "name": "torch_fused_softmax_std_hardtanh_bf16",
    "inputs": [
        ((4, 10), torch.float32),
    ],
    "outputs": [
        ((4, 1), torch.float32),
    ]
}
