
import torch

def torch_box_filter_svd(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Applies a box filter to an input tensor, followed by SVD decomposition.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        kernel_size (int): Size of the box filter kernel.

    Returns:
        torch.Tensor: The singular values of the filtered tensor, shape (B, C, 1, 1).
    """

    # Apply box filter
    filtered_tensor = torch.nn.functional.avg_pool2d(input_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    # Compute SVD
    U, S, V = torch.linalg.svd(filtered_tensor.view(input_tensor.shape[0], input_tensor.shape[1], -1))

    # Extract singular values
    return S.unsqueeze(-1).unsqueeze(-1)


function_signature = {
    "name": "torch_box_filter_svd",
    "inputs": [
        ((2, 3, 4, 4), torch.float32),
        (3, torch.int32)
    ],
    "outputs": [
        ((2, 3, 1, 1), torch.float32),
    ]
}
