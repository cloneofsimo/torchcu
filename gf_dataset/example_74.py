
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

def torch_edge_detection_with_batch_norm(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Performs edge detection using Canny edge detection followed by batch normalization,
    all in fp16 precision.
    """
    input_fp16 = input_tensor.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)
    bias_fp16 = bias.to(torch.float16)
    running_mean_fp16 = running_mean.to(torch.float16)
    running_var_fp16 = running_var.to(torch.float16)

    # Canny edge detection
    edges = torch.nn.functional.canny_edge_detection(input_fp16, 0.1, 0.2)

    # Batch normalization
    output_fp16 = F.batch_norm(edges, running_mean_fp16, running_var_fp16, weight_fp16, bias_fp16, training=False, eps=eps)
    return output_fp16.to(torch.float32)

function_signature = {
    "name": "torch_edge_detection_with_batch_norm",
    "inputs": [
        ((1, 1, 128, 128), torch.float32),
        ((1, 1), torch.float32),
        ((1,), torch.float32),
        ((1,), torch.float32),
        ((1,), torch.float32),
    ],
    "outputs": [
        ((1, 1, 128, 128), torch.float32),
    ]
}
