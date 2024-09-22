
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def torch_drop_path_logsigmoid_mixing_fp16(input_tensor: torch.Tensor, weight: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """
    Applies drop path with probability `drop_prob` and logsigmoid activation to the input tensor.
    Then performs feature mixing with a given weight tensor.
    """
    if drop_prob > 0.0 and self.training:
        keep_prob = 1.0 - drop_prob
        mask = (torch.rand(input_tensor.shape[0], 1, 1, 1, device=input_tensor.device) < keep_prob).float()
        output = input_tensor * mask / keep_prob
    else:
        output = input_tensor

    output = F.logsigmoid(output)

    with autocast():
        output = torch.einsum('bchw,io->bcio', output, weight)

    return output.to(torch.float32)

function_signature = {
    "name": "torch_drop_path_logsigmoid_mixing_fp16",
    "inputs": [
        ((1, 4, 4, 4), torch.float32),
        ((4, 8), torch.float32),
        (0.1, torch.float32),
    ],
    "outputs": [
        ((1, 8, 4, 4), torch.float32)
    ]
}
