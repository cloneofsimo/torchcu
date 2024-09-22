
import torch
import torch.nn.functional as F

def torch_ctc_loss_fp16(input_tensor: torch.Tensor, target_tensor: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
    """
    Calculate CTC loss with fp16 precision for better performance.
    """
    input_tensor = input_tensor.to(torch.float16)
    target_tensor = target_tensor.to(torch.int32)
    input_lengths = input_lengths.to(torch.int32)
    target_lengths = target_lengths.to(torch.int32)
    loss = F.ctc_loss(log_probs=input_tensor, targets=target_tensor, input_lengths=input_lengths, target_lengths=target_lengths, blank=0)
    return loss.to(torch.float32)

function_signature = {
    "name": "torch_ctc_loss_fp16",
    "inputs": [
        ((10, 32, 10), torch.float32),
        ((10,), torch.int32),
        ((10,), torch.int32),
        ((10,), torch.int32),
    ],
    "outputs": [
        ((), torch.float32),
    ]
}
