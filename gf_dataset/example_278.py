
import torch
import torch.nn.functional as F

def torch_adaptive_avg_pool_with_label_smoothing(input_tensor: torch.Tensor, target: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """
    Applies adaptive average pooling to the input tensor, followed by label smoothing on the target.
    """
    output = F.adaptive_avg_pool1d(input_tensor, 1)
    output = output.squeeze(dim=-1)  # Squeeze to remove the singleton dimension

    # Label smoothing
    with torch.no_grad():
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - smoothing) + smoothing / output.size(1)
    
    loss = F.mse_loss(output, smooth_one_hot)
    loss.backward()
    return loss

function_signature = {
    "name": "torch_adaptive_avg_pool_with_label_smoothing",
    "inputs": [
        ((1, 10, 20), torch.float32),
        ((1,), torch.int64)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
