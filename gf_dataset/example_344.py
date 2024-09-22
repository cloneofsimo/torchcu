
import torch
import torch.nn.functional as F
from torch.quantization import quantize_dynamic

def torch_quantized_conv2d(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a quantized 2D convolution with dynamic quantization.
    """
    # Quantize the model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0], kernel_size=weight.shape[2:], bias=False),
    )
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model.fuse_in_batch_norm(weight, bias)
    model = quantize_dynamic(model, qconfig=model.qconfig)

    # Perform the convolution
    output = model(input_tensor)
    return output

function_signature = {
    "name": "torch_quantized_conv2d",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((16, 3, 3, 3), torch.float32),
        ((16,), torch.float32)
    ],
    "outputs": [
        ((1, 16, 224, 224), torch.float32),
    ]
}
