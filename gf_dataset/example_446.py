
import torch

def torch_softmax_temperature_function(input_tensor: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply softmax with temperature to input tensor.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = torch.nn.functional.softmax(input_bf16 / temperature, dim=1, dtype=torch.bfloat16)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_softmax_temperature_function",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
