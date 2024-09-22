
import torch
import torch.nn.functional as F

def torch_transposed_conv3d_hinge_loss_bf16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Performs a transposed 3D convolution, applies a hinge embedding loss, and returns the loss value in bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)
    output_bf16 = F.conv_transpose3d(input_bf16, weight_bf16, bias_bf16, stride=2, padding=1, output_padding=1)
    loss_bf16 = F.hinge_embedding_loss(output_bf16, target.to(torch.bfloat16))
    return loss_bf16.to(torch.float32)

function_signature = {
    "name": "torch_transposed_conv3d_hinge_loss_bf16",
    "inputs": [
        ((2, 3, 4, 5, 6), torch.float32),
        ((3, 2, 3, 3, 3), torch.float32),
        ((3,), torch.float32),
        ((2, 3, 8, 10, 12), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
