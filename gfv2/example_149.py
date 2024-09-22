
import torch

def int8_add_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor) -> torch.Tensor:
    """
    Perform element-wise addition using int8 data type.
    """
    input1_int8 = input_tensor1.to(torch.int8)
    input2_int8 = input_tensor2.to(torch.int8)
    output = input1_int8 + input2_int8
    return output.to(torch.int32)

function_signature = {
    "name": "int8_add_function",
    "inputs": [
        ((4, 4), torch.int32),
        ((4, 4), torch.int32)
    ],
    "outputs": [
        ((4, 4), torch.int32)
    ]
}
