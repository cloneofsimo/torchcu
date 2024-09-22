
import torch

def fused_dropout_relu_function(input_tensor: torch.Tensor, p: float) -> torch.Tensor:
    """
    Applies fused dropout and ReLU activation.
    """
    output = torch.nn.functional.dropout(input_tensor, p=p, training=True, inplace=False)
    return torch.relu(output)

function_signature = {
    "name": "fused_dropout_relu_function",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
