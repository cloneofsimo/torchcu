
import torch

def torch_hardtanh_unique_fp16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies hardtanh activation to an input tensor, then finds the unique elements and returns them in fp16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    output = torch.hardtanh(input_fp16, min_val=-1.0, max_val=1.0)
    unique_values = torch.unique(output)
    return unique_values.to(torch.float16)

function_signature = {
    "name": "torch_hardtanh_unique_fp16_function",
    "inputs": [
        ((10,), torch.float32)
    ],
    "outputs": [
        ((None,), torch.float16),
    ]
}
