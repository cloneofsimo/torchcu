
import torch
import torch.nn.functional as F

def torch_laplace_filter_bf16_function(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Applies a Laplacian filter to an input tensor using bfloat16 for improved performance.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.bfloat16)
    output_bf16 = F.conv2d(input_bf16.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size//2)
    return output_bf16.squeeze(0).squeeze(0).to(torch.float32)

function_signature = {
    "name": "torch_laplace_filter_bf16_function",
    "inputs": [
        ((4, 4), torch.float32),
        (1, torch.int32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
