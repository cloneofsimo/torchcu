
import torch

def pruned_qr_decomposition(input_tensor: torch.Tensor, pruning_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs QR decomposition with element-wise pruning based on a mask.

    Args:
        input_tensor: The input tensor to decompose.
        pruning_mask: A boolean tensor indicating which elements to keep.

    Returns:
        The Q matrix from the QR decomposition, with pruned elements set to zero.
    """
    # Apply pruning mask
    masked_tensor = input_tensor * pruning_mask

    # Perform QR decomposition on the pruned tensor
    q, r = torch.linalg.qr(masked_tensor)

    # Apply the mask to the Q matrix (elements corresponding to pruned elements are set to zero)
    q_pruned = q * pruning_mask.unsqueeze(dim=1).expand_as(q)

    return q_pruned

function_signature = {
    "name": "pruned_qr_decomposition",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.bool),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
