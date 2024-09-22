
import torch

def avg_pool2d_int8_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs average pooling in 2D with int8 data type. 
    """
    input_int8 = input_tensor.to(torch.int8)
    output = torch.nn.functional.avg_pool2d(input_int8, kernel_size=2, stride=2)
    return output.to(torch.float32)

function_signature = {
    "name": "avg_pool2d_int8_function",
    "inputs": [
        ((4, 3, 8, 8), torch.float32)
    ],
    "outputs": [
        ((4, 3, 4, 4), torch.float32),
    ]
}

