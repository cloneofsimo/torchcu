
import torch

def label_smoothing_int8(input_tensor: torch.Tensor, labels: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """
    Applies label smoothing to the input tensor and returns the smoothed labels.
    """
    confidence = 1.0 - smoothing
    log_probs = torch.log_softmax(input_tensor, dim=1)
    nll_loss = -log_probs.gather(dim=1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -log_probs.mean(dim=1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.to(torch.int8)

function_signature = {
    "name": "label_smoothing_int8",
    "inputs": [
        ((10, 5), torch.float32),
        ((10,), torch.int64)
    ],
    "outputs": [
        ((10,), torch.int8)
    ]
}
