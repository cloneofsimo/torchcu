
import torch

def torch_uniform_fp16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies a uniform distribution to the input tensor, seeds the generator, and then
    returns the result in fp16.
    """
    torch.manual_seed(42)
    output = torch.nn.functional.uniform(input_tensor, -1.0, 1.0)  # Uniform distribution
    return output.to(torch.float16)

function_signature = {
    "name": "torch_uniform_fp16_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float16),
    ]
}
