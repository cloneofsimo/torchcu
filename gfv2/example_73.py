
import torch
import torch.nn.functional as F

def feature_extractor(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Applies a series of operations to extract features from an input tensor.

    1. Applies average pooling with a kernel size of 3.
    2. Calculates pairwise Chebyshev distance between the pooled features and a weight matrix.
    3. Applies a linear transformation with the provided weight and bias.
    """
    # Average pooling
    pooled = F.avg_pool2d(input_tensor, kernel_size=3)

    # Pairwise Chebyshev distance
    distances = torch.cdist(pooled.flatten(1), weight.flatten(1), p=float('inf'))

    # Linear transformation
    output = F.linear(distances, weight, bias)

    return output.to(torch.float16)  # Return output in fp16

function_signature = {
    "name": "feature_extractor",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((1024, 9), torch.float32),
        ((1024,), torch.float32)
    ],
    "outputs": [
        ((1, 1024), torch.float16),
    ]
}
