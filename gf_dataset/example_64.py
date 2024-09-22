
import torch

def torch_topk_irfft_sigmoid_fp16(input_tensor: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    """
    Applies top-k, inverse real-to-complex FFT, and sigmoid activation to an input tensor.
    """
    input_fp16 = input_tensor.to(torch.float16)
    topk_values, _ = torch.topk(input_fp16, k, dim=dim)
    irfft_output = torch.irfft(topk_values, signal_ndim=1, normalized=True)
    sigmoid_output = torch.sigmoid(irfft_output)
    return sigmoid_output.to(torch.float32)


function_signature = {
    "name": "torch_topk_irfft_sigmoid_fp16",
    "inputs": [
        ((4, 4, 8), torch.float32),
        (1, torch.int32),
        (1, torch.int32),
    ],
    "outputs": [
        ((4, 4, 4), torch.float32),
    ]
}
