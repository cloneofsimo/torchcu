
import torch
import torch.nn.functional as F

def torch_roberts_cross_gradient_dropout_bf16(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies Roberts Cross Gradient filter, then fused dropout, and returns the result in bfloat16.
    """
    # Roberts Cross Gradient
    kernel = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)
    gradient_x = F.conv2d(input_tensor, kernel, padding=1)
    kernel = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)
    gradient_y = F.conv2d(input_tensor, kernel, padding=1)
    gradient = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Fused Dropout
    gradient = F.dropout(gradient, p=0.5, training=True)  # Assume training

    # Return in bfloat16
    return gradient.to(torch.bfloat16)


function_signature = {
    "name": "torch_roberts_cross_gradient_dropout_bf16",
    "inputs": [
        ((3, 3, 10, 10), torch.float32)
    ],
    "outputs": [
        ((3, 3, 10, 10), torch.bfloat16),
    ]
}
