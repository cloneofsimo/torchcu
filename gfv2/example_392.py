
import torch

def complex_function(input_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor:
    1. Applies a soft shrink function.
    2. Calculates the mean of the shrunk tensor.
    3. Creates an empty tensor with the same shape as the input.
    4. Fills the empty tensor with the calculated mean value.
    5. Returns the filled tensor.
    """
    shrunk_tensor = torch.nn.functional.softshrink(input_tensor, lambd=threshold)
    mean_value = shrunk_tensor.mean()
    output_tensor = torch.empty_like(input_tensor)
    output_tensor.fill_(mean_value)
    return output_tensor

function_signature = {
    "name": "complex_function",
    "inputs": [
        ((1,), torch.float32),
        (torch.float32,),
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
