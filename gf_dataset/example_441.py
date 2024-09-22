
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def time_stretch_orthogonal_regularization_bf16(input_tensor: torch.Tensor, stretch_factor: float, regularization_weight: float) -> torch.Tensor:
    """
    Applies time stretching to an input tensor and adds orthogonal regularization to the stretched output.

    Args:
        input_tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length, features).
        stretch_factor (float): The factor by which to stretch the sequence in time.
        regularization_weight (float): The weight of the orthogonal regularization loss.

    Returns:
        torch.Tensor: The stretched and regularized output tensor.
    """

    # Convert to bfloat16
    input_bf16 = input_tensor.to(torch.bfloat16)

    # Time stretching
    stretched_tensor = F.interpolate(input_bf16, scale_factor=stretch_factor, mode='linear', align_corners=False)

    # Orthogonal regularization
    orthogonal_loss = 0.0
    for i in range(stretched_tensor.size(1)):
        W = stretched_tensor[:, i, :].T
        orthogonal_loss += torch.norm(torch.matmul(W, W.T) - torch.eye(W.size(0)))

    # Combine loss with original output
    output = stretched_tensor.to(torch.float32) + orthogonal_loss * regularization_weight

    return output

function_signature = {
    "name": "time_stretch_orthogonal_regularization_bf16",
    "inputs": [
        ((8, 16, 32), torch.float32),
        (torch.float32,),
        (torch.float32,)
    ],
    "outputs": [
        ((8, int(16 * stretch_factor), 32), torch.float32)
    ]
}
