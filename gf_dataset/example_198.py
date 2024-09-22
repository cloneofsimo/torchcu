
import torch
from torch.nn.functional import softmax
from torch.cuda.amp import custom_bwd

@custom_bwd
def torch_masked_softmax_softplus(input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies softmax to the input tensor along the last dimension,
    masks out values according to the attention mask, and then applies softplus.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).

    Returns:
        torch.Tensor: Output tensor of the same shape as input_tensor.
    """
    masked_input = input_tensor * attention_mask.unsqueeze(-1)
    softmax_output = softmax(masked_input, dim=-1)
    return torch.nn.functional.softplus(softmax_output)

function_signature = {
    "name": "torch_masked_softmax_softplus",
    "inputs": [
        ((16, 128, 768), torch.float32),
        ((16, 128), torch.bool)
    ],
    "outputs": [
        ((16, 128, 768), torch.float32)
    ]
}
