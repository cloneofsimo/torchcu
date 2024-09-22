
import torch

def logsigmoid_gather_fp16(input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Applies logsigmoid activation to input tensor, gathers values based on indices,
    and returns the result in fp16.
    """
    input_tensor_fp16 = input_tensor.to(torch.float16)
    logsigmoid_output = torch.logsigmoid(input_tensor_fp16)
    gathered_output = torch.gather(logsigmoid_output, dim=1, index=indices)
    return gathered_output.to(torch.float16)

function_signature = {
    "name": "logsigmoid_gather_fp16",
    "inputs": [
        ((10, 5), torch.float32),
        ((10, 1), torch.int8)
    ],
    "outputs": [
        ((10, 1), torch.float16),
    ]
}
