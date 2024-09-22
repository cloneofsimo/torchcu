
import torch

def torch_nll_loss_round_fp16(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the negative log likelihood loss, rounds the output, and converts it to fp16.
    """
    loss = torch.nn.functional.nll_loss(input_tensor, target)
    rounded_loss = torch.round(loss)
    return rounded_loss.to(torch.float16)

function_signature = {
    "name": "torch_nll_loss_round_fp16",
    "inputs": [
        ((10,), torch.float32),
        ((10,), torch.long)
    ],
    "outputs": [
        ((), torch.float16),
    ]
}
