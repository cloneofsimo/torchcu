
import torch

def complex_function(input_tensor: torch.Tensor, weights: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Performs a series of operations on the input tensor and returns a list of tensors.
    """
    output_tensors = []
    for i, weight in enumerate(weights):
        output = torch.matmul(input_tensor, weight.t())
        output = torch.sigmoid(output)
        output_tensors.append(output)
    return output_tensors

function_signature = {
    "name": "complex_function",
    "inputs": [
        ((4, 4), torch.float32),
        [((4, 4), torch.float32), ((4, 4), torch.float32)]
    ],
    "outputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ]
}
