
import torch

def torch_qr_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs QR decomposition on the input tensor using the 'qr' function.
    """
    q, r = torch.linalg.qr(input_tensor)
    return q

function_signature = {
    "name": "torch_qr_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
