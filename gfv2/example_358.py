
import torch

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Calculates L1 loss between input_tensor and weight, applies Mish activation, 
    and returns the result in FP32.
    """
    input_tensor_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    loss = torch.abs(input_tensor_int8 - weight_int8)
    output = torch.mish(loss.float())
    return output.float()

function_signature = {
    "name": "my_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
