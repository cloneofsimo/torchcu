
import torch
import torch.nn.functional as F

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor, dropout_p: float) -> torch.Tensor:
    """
    Performs a linear transformation, applies fused dropout, tanh activation, and returns the result in fp32.
    """
    output = torch.matmul(input_tensor, weight.t())
    output = F.dropout(output, p=dropout_p, training=True, inplace=True)  # Fused dropout
    output = torch.tanh(output)
    return output.to(torch.float32)

function_signature = {
    "name": "my_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        (float, )
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
