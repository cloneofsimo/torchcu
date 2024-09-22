
import torch

def torch_einsum_dropout_function(input_tensor: torch.Tensor, weight: torch.Tensor, dropout_p: float) -> torch.Tensor:
    """
    Performs a batched einsum contraction, applies dropout, and returns the result.
    """
    output = torch.einsum('bni,ijk->bnk', input_tensor, weight)
    output = torch.nn.functional.dropout(output, p=dropout_p, inplace=True)
    return output

function_signature = {
    "name": "torch_einsum_dropout_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((4, 5, 6), torch.float32),
        (0.2, torch.float32)
    ],
    "outputs": [
        ((2, 3, 6), torch.float32),
    ]
}
