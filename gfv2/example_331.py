
import torch

def adaptive_avg_pool1d_int8_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs adaptive average pooling in 1D, converting the input to int8 and returning the result as fp32.
    """
    input_int8 = input_tensor.to(torch.int8)
    output = torch.nn.functional.adaptive_avg_pool1d(input_int8, output_size=1)
    return output.to(torch.float32)

function_signature = {
    "name": "adaptive_avg_pool1d_int8_function",
    "inputs": [
        ((1, 1, 10), torch.float32)
    ],
    "outputs": [
        ((1, 1, 1), torch.float32),
    ]
}
