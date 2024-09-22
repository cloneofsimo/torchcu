
import torch

def elementwise_div_fp16(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise division between two tensors, using fp16 for computation. 
    """
    input1_fp16 = input_tensor1.to(torch.float16)
    input2_fp16 = input_tensor2.to(torch.float16)
    output = torch.div(input1_fp16, input2_fp16)
    return output.to(torch.float32)


function_signature = {
    "name": "elementwise_div_fp16",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
