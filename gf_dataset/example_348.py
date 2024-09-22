
import torch
import torch.nn.functional as F

def torch_avgpool3d_inplace(input_tensor: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Perform 3D average pooling with inplace modification.
    """
    return F.avg_pool3d(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, inplace=True)

function_signature = {
    "name": "torch_avgpool3d_inplace",
    "inputs": [
        ((2, 3, 4, 5, 6), torch.float32),
        (3,), torch.int32,
        (3,), torch.int32,
        (3,), torch.int32
    ],
    "outputs": [
        ((2, 3, 2, 3, 4), torch.float32),
    ]
}
