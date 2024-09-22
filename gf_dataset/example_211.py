
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.cuda import cutlass

class CTCLossWrapper(torch.nn.Module):
    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        log_probs = log_probs.transpose(0, 1)  # [T, N, C] -> [N, T, C]
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                          blank=self.blank, reduction=self.reduction,
                          zero_infinity=self.zero_infinity)

def torch_ctc_loss_with_padding(log_probs: torch.Tensor, targets: torch.Tensor,
                                 input_lengths: torch.Tensor, target_lengths: torch.Tensor,
                                 blank: int = 0) -> torch.Tensor:
    """
    Compute CTC loss with padding.

    Args:
        log_probs: Log probabilities of each character at each time step, shape [N, T, C]
        targets: Target labels, shape [N, T_max]
        input_lengths: Length of each input sequence, shape [N]
        target_lengths: Length of each target sequence, shape [N]
        blank: Index of blank symbol

    Returns:
        CTC loss, shape [1]
    """
    targets = pad_sequence(targets, batch_first=True, padding_value=blank)
    loss_fn = CTCLossWrapper(blank=blank, reduction='mean')
    return loss_fn(log_probs, targets, input_lengths, target_lengths)

function_signature = {
    "name": "torch_ctc_loss_with_padding",
    "inputs": [
        ((16, 32, 50), torch.float32),
        ((16, 10), torch.int64),
        ((16,), torch.int64),
        ((16,), torch.int64),
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
