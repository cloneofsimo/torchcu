
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

@custom_fwd(cast_inputs=torch.bfloat16)
def forward(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """
    Performs Hadamard product, pairwise Chebyshev distance, sigmoid focal loss, and returns the result.
    """
    # Hadamard product
    hadamard_product = input1 * input2

    # Pairwise Chebyshev distance
    chebyshev_distance = torch.max(torch.abs(input1 - input2), dim=1).values

    # Sigmoid focal loss
    sigmoid_focal_loss = F.binary_cross_entropy_with_logits(input1, input2, reduction='none') * (1 - input2)**2

    # Combine all outputs
    output = torch.stack([hadamard_product, chebyshev_distance, sigmoid_focal_loss], dim=1)
    return output

function_signature = {
    "name": "forward",
    "inputs": [
        ((10, 10), torch.float32),
        ((10, 10), torch.float32)
    ],
    "outputs": [
        ((10, 3), torch.float32),
    ]
}
