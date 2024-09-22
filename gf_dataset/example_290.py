
import torch
from torch.cuda import amp

def torch_tanh_eq_zcr_function(input_tensor: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies tanh activation, compares with threshold, and computes zero-crossing rate.
    """
    with amp.autocast():  # Use automatic mixed precision for potential speedup
        output_tanh = torch.tanh(input_tensor)
        output_eq = torch.eq(output_tanh, threshold)
        
        # Calculate zero-crossing rate (ZCR)
        zcr = torch.zeros_like(input_tensor, dtype=torch.float32)
        for i in range(1, input_tensor.shape[1]):
            zcr[:, i] = ((output_tanh[:, i] * output_tanh[:, i - 1]) < 0).float()
        zcr = torch.mean(zcr, dim=1)
    
    return output_tanh, zcr

function_signature = {
    "name": "torch_tanh_eq_zcr_function",
    "inputs": [
        ((16, 1024), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((16, 1024), torch.float32),
        ((16,), torch.float32)
    ]
}
