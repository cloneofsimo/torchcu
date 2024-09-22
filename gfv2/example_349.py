
import torch

def padded_topk_gradient_clipping(input_tensor: torch.Tensor, k: int, padding_value: float, max_norm: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies padding, performs top-k selection, and applies gradient clipping to the input tensor.

    Args:
        input_tensor: The input tensor.
        k: The number of top elements to select.
        padding_value: The value to use for padding.
        max_norm: The maximum norm of the gradient to clip.

    Returns:
        A tuple containing:
            - The padded and top-k selected tensor.
            - The indices of the selected elements.
    """

    # Pad the input tensor
    padded_tensor = torch.nn.functional.pad(input_tensor, (0, 0, 0, k - input_tensor.size(1)), "constant", padding_value)

    # Perform top-k selection
    topk_values, topk_indices = torch.topk(padded_tensor, k, dim=1)

    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(topk_values, max_norm)

    return topk_values.to(torch.float32), topk_indices

function_signature = {
    "name": "padded_topk_gradient_clipping",
    "inputs": [
        ((1, 4), torch.float32),
        (int,),
        (float,),
        (float,)
    ],
    "outputs": [
        ((1, 4), torch.float32),
        ((1, 4), torch.int64)
    ]
}
