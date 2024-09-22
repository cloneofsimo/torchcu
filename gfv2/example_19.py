
import torch
import torch.nn.functional as F

def gumbel_softmax_hardsigmoid_envelope_threshold(input_tensor: torch.Tensor, weights: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Applies Gumbel-Softmax, hardsigmoid, signal envelope, and thresholding to an input tensor.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape (batch_size, features).
        weights (torch.Tensor): Weights tensor with shape (features, output_features).
        threshold (float): Threshold value for the final output.

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, output_features) after applying all operations.
    """

    # Gumbel-Softmax
    gumbel_noise = torch.rand_like(input_tensor)
    gumbel_noise = -torch.log(-torch.log(gumbel_noise))
    gumbel_output = F.softmax((input_tensor + gumbel_noise) / 1.0, dim=-1)  # Temperature set to 1.0

    # Hardsigmoid
    hardsigmoid_output = F.hardsigmoid(torch.matmul(gumbel_output, weights))

    # Signal Envelope
    envelope_output = torch.abs(hardsigmoid_output)

    # Thresholding
    threshold_output = (envelope_output > threshold).float()

    return threshold_output

function_signature = {
    "name": "gumbel_softmax_hardsigmoid_envelope_threshold",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
