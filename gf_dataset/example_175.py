
import torch

def torch_softmax_temperature_fp16_function(input_tensor: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Applies softmax with temperature scaling to the input tensor.
    """
    input_tensor_fp16 = input_tensor.to(torch.float16)
    output_fp16 = torch.softmax(input_tensor_fp16 / temperature, dim=-1)
    return output_fp16.to(torch.float16)

function_signature = {
    "name": "torch_softmax_temperature_fp16_function",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32,)
    ],
    "outputs": [
        ((4, 4), torch.float16),
    ]
}
