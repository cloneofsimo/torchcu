
import torch

def fused_operation(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, label: torch.Tensor) -> list[torch.Tensor]:
    """
    Performs a series of fused operations:
    1. Adaptive average pooling of the input tensor
    2. Matrix multiplication with the weight tensor
    3. Add bias and apply label smoothing
    4. Scale the output using addcmul
    """
    # Adaptive Average Pooling (2D)
    pooled = torch.nn.functional.adaptive_avg_pool2d(input_tensor.to(torch.bfloat16), (1, 1)).to(torch.float32)

    # Matrix Multiplication (with bfloat16 precision)
    weight_bf16 = weight.to(torch.bfloat16)
    output = torch.matmul(pooled, weight_bf16.t()).to(torch.float32)

    # Add bias and Label Smoothing
    label_smooth = (1 - 0.1) * label + 0.1 / label.size(1)  # Example label smoothing with factor 0.1
    output += bias + label_smooth

    # Addcmul (with int8 precision)
    scale = torch.tensor(1.0, dtype=torch.int8)
    output.addcmul_(scale, input_tensor.to(torch.int8), weight.to(torch.int8))  # In-place operation

    return [output, label_smooth]

function_signature = {
    "name": "fused_operation",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((1000, 3), torch.float32),
        ((1000,), torch.float32),
        ((1000,), torch.float32),
    ],
    "outputs": [
        ((1, 1000), torch.float32),
        ((1, 1000), torch.float32),
    ]
}
