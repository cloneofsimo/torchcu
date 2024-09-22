
import torch

def torch_elementwise_div_fp16_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise division using fp16.
    """
    input_tensor1_fp16 = input_tensor1.to(torch.float16)
    input_tensor2_fp16 = input_tensor2.to(torch.float16)
    output = torch.div(input_tensor1_fp16, input_tensor2_fp16)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_elementwise_div_fp16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
