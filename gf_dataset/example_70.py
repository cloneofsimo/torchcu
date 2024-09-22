
import torch

def torch_multi_margin_loss_int8_function(input_tensor: torch.Tensor, target: torch.Tensor, p: float = 1.0, margin: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
    """
    Calculate the Multi-margin loss with int8 precision.
    """
    input_tensor_int8 = input_tensor.to(torch.int8)
    target_int8 = target.to(torch.int8)
    loss = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=reduction)
    output = loss(input_tensor_int8, target_int8)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_multi_margin_loss_int8_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4,), torch.int64),
        (1.0, torch.float32),
        (1.0, torch.float32),
        ('mean', torch.str)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
