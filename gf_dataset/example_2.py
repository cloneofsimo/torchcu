
import torch

def torch_relu_ge_cudnn_function(input_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Applies the ReLU function with a given threshold using cuDNN.
    """
    return torch.nn.functional.relu(input_tensor, inplace=False, threshold=threshold)

function_signature = {
    "name": "torch_relu_ge_cudnn_function",
    "inputs": [
        ((4, 4), torch.float32),
        (float, torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
