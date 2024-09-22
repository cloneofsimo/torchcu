
import torch
import torch.nn.functional as F

def my_complex_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations, including:
    1. Linear transformation with given weight.
    2. Applies sigmoid activation.
    3. Calculates the NLL loss with the target tensor.
    4. Calculates the Frobenius norm of the input tensor.
    5. Applies ceil function to the Frobenius norm and converts it to int8.
    6. Returns the NLL loss, Frobenius norm, and the ceiled norm in a list.
    """
    output = torch.matmul(input_tensor, weight.t())
    output = torch.sigmoid(output)
    nll_loss = F.nll_loss(output, target_tensor)
    frobenius_norm = torch.linalg.norm(input_tensor, ord="fro")
    ceiled_norm = torch.ceil(frobenius_norm).to(torch.int8)
    return [nll_loss, frobenius_norm, ceiled_norm]

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4,), torch.int64),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
        ((1,), torch.float32),
        ((1,), torch.int8)
    ]
}
