
import torch
import torch.nn.functional as F

def torch_pre_activation_sum_fp16(input1: torch.Tensor, input2: torch.Tensor, input3: torch.Tensor) -> torch.Tensor:
    """
    Performs a pre-activation operation, sum, and returns the result in fp16.
    """
    input1_fp16 = input1.to(torch.float16)
    input2_fp16 = input2.to(torch.float16)
    input3_fp16 = input3.to(torch.float16)
    
    pre_activated = F.relu(input1_fp16) + input2_fp16
    output = pre_activated + input3_fp16
    return output.to(torch.float16)

function_signature = {
    "name": "torch_pre_activation_sum_fp16",
    "inputs": [
        ((8, 8, 8), torch.float32),
        ((8, 8, 8), torch.float32),
        ((8, 8, 8), torch.float32)
    ],
    "outputs": [
        ((8, 8, 8), torch.float16),
    ]
}
