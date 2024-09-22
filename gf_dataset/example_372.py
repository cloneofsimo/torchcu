
import torch

def torch_feature_mixing_fp16(input_tensor: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor) -> torch.Tensor:
    """
    Mixes features from two different weights and returns the result in FP16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    weight1_fp16 = weight1.to(torch.float16)
    weight2_fp16 = weight2.to(torch.float16)

    mixed_features = torch.matmul(input_fp16, weight1_fp16.t()) + torch.matmul(input_fp16, weight2_fp16.t())
    return mixed_features.to(torch.float16)

function_signature = {
    "name": "torch_feature_mixing_fp16",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float16)
    ]
}
