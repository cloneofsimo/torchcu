
import torch

def my_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies average pooling (kernel_size=3, stride=2) on the input tensor and returns the result in fp16.
    """
    output = torch.nn.functional.avg_pool1d(input_tensor.unsqueeze(1), kernel_size=3, stride=2)
    output = output.squeeze(1).to(torch.float16)
    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((10, 4), torch.float32),
    ],
    "outputs": [
        ((5, 4), torch.float16),
    ]
}
