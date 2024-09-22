
import torch

def conv2d_sigmoid_focal_loss(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Performs a 2D convolution, applies sigmoid activation, and calculates sigmoid focal loss.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    output = torch.nn.functional.conv2d(input_bf16, weight_bf16, bias_bf16)
    output = torch.sigmoid(output).to(torch.float32)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target, reduction='none')
    loss = (1 - output) ** 2 * loss

    return loss

function_signature = {
    "name": "conv2d_sigmoid_focal_loss",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((3, 3, 3, 3), torch.float32),
        ((3,), torch.float32),
        ((1, 1, 224, 224), torch.float32)
    ],
    "outputs": [
        ((1, 1, 224, 224), torch.float32),
    ]
}
