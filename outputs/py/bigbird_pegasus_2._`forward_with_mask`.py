import torch

def forward_with_mask(inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a mask to the inputs before passing them through a forward pass.

    Args:
        inputs (torch.Tensor): The input tensor.
        attention_mask (torch.Tensor): The attention mask tensor.

    Returns:
        torch.Tensor: The masked input tensor.
    """
    return inputs * attention_mask.unsqueeze(-1)



# function_signature
function_signature = {
    "name": "forward_with_mask",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 1), torch.float32)
    ],
    "outputs": [((4, 4, 4), torch.float32)]
}