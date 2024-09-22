
import torch

def torch_mean_adaptive_avg_pool_bfloat16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs adaptive average pooling followed by mean calculation, using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = torch.mean(torch.nn.AdaptiveAvgPool2d((1, 1))(input_bf16), dim=(2, 3))
    return output.to(torch.float32)

function_signature = {
    "name": "torch_mean_adaptive_avg_pool_bfloat16_function",
    "inputs": [
        ((1, 3, 224, 224), torch.float32)
    ],
    "outputs": [
        ((1, 3), torch.float32)
    ]
}
