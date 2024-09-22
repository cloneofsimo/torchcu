
import torch
import torch.nn.functional as F

def torch_rrelu_pool_log_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform RReLU activation, adaptive average pooling, log operation, and then compute backward.
    """
    input_fp16 = input_tensor.to(torch.float16)
    output = F.rrelu(input_fp16, lower=0.125, upper=0.334)
    output = F.adaptive_avg_pool2d(output, (2, 2))
    output = torch.log(output)
    output.backward(torch.ones_like(output))
    return output.to(torch.float32)

function_signature = {
    "name": "torch_rrelu_pool_log_function",
    "inputs": [
        ((1, 3, 8, 8), torch.float32),
    ],
    "outputs": [
        ((1, 3, 2, 2), torch.float32),
    ]
}
