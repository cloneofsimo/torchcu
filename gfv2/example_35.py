
import torch
import torch.nn.functional as F

def exponential_nll_loss_fp16(input_tensor: torch.Tensor, target_tensor: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Applies exponential function to the input tensor, calculates the negative log likelihood loss with the target,
    and returns the loss in fp16 precision.
    """
    input_tensor_fp16 = input_tensor.to(torch.float16)
    output = torch.exp(input_tensor_fp16)
    loss = F.nll_loss(torch.log(output), target_tensor, reduction=reduction)
    return loss.to(torch.float16)

function_signature = {
    "name": "exponential_nll_loss_fp16",
    "inputs": [
        ((4, 4), torch.float32),
        ((4,), torch.long)
    ],
    "outputs": [
        ((1,), torch.float16),
    ]
}
