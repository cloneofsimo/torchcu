
import torch

def elementwise_min_fp16(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor) -> torch.Tensor:
    """
    Perform element-wise minimum operation on two tensors in fp16.
    """
    input_tensor1_fp16 = input_tensor1.to(torch.float16)
    input_tensor2_fp16 = input_tensor2.to(torch.float16)
    output = torch.min(input_tensor1_fp16, input_tensor2_fp16)
    return output.to(torch.float32)

function_signature = {
    "name": "elementwise_min_fp16",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
