
import torch
import torch.nn.functional as F

def multi_margin_loss_with_padding(input_tensor: torch.Tensor, target: int, weight: torch.Tensor,
                                  padding: int = 1, pad_value: float = 0.0) -> torch.Tensor:
    """
    Applies multi-margin loss with constant padding to the input tensor.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        target (int): The target class index.
        weight (torch.Tensor): The weight tensor.
        padding (int, optional): The padding size. Defaults to 1.
        pad_value (float, optional): The padding value. Defaults to 0.0.

    Returns:
        torch.Tensor: The computed loss.
    """
    # Pad the input tensor
    padded_input = F.pad(input_tensor, (padding, padding), "constant", value=pad_value)

    # Apply multi-margin loss
    loss = F.multi_margin_loss(padded_input, torch.tensor([target]), weight=weight)

    return loss

function_signature = {
    "name": "multi_margin_loss_with_padding",
    "inputs": [
        ((1,), torch.float32),
        (torch.int32, ),
        ((1,), torch.float32),
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
