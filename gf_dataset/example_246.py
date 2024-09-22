
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

@custom_fwd(cast_inputs=torch.bfloat16)
def laplace_filter_bfloat16(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies a 2D Laplacian filter to a tensor.
    """
    kernel = torch.tensor([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=torch.bfloat16, device=input_tensor.device)
    output = F.conv2d(input_tensor.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0), padding=1)
    return output.squeeze(1).to(torch.float32)

function_signature = {
    "name": "laplace_filter_bfloat16",
    "inputs": [
        ((1, 10, 10), torch.float32)
    ],
    "outputs": [
        ((1, 10, 10), torch.float32)
    ]
}
