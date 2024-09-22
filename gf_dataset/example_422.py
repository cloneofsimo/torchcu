
import torch
import torch.nn.functional as F

def torch_attention_gradient_function(input_tensor: torch.Tensor, window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a windowed attention mechanism, calculates the gradient using Prewitt kernel, and returns both.
    """

    # Windowed Attention
    B, C, H, W = input_tensor.size()
    attention_weights = F.softmax(input_tensor.view(B, C, H // window_size, window_size, W // window_size, window_size)
                                  .mean(dim=(3, 5)), dim=2)
    attention_output = (input_tensor.view(B, C, H // window_size, window_size, W // window_size, window_size)
                       * attention_weights.unsqueeze(1).unsqueeze(1))
    attention_output = attention_output.view(B, C, H, W)

    # Prewitt Gradient
    kernel_x = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32, device=input_tensor.device)
    kernel_y = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32, device=input_tensor.device)
    gradient_x = F.conv2d(attention_output, kernel_x.unsqueeze(0).unsqueeze(0), padding=1)
    gradient_y = F.conv2d(attention_output, kernel_y.unsqueeze(0).unsqueeze(0), padding=1)

    return attention_output, torch.stack((gradient_x, gradient_y), dim=1)

function_signature = {
    "name": "torch_attention_gradient_function",
    "inputs": [
        ((1, 3, 24, 24), torch.float32),
        ((), torch.int32)
    ],
    "outputs": [
        ((1, 3, 24, 24), torch.float32),
        ((1, 2, 24, 24), torch.float32)
    ]
}
