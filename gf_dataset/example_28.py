
import torch
import torch.nn.functional as F

def torch_multi_label_margin_loss_with_softmax(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the multi-label margin loss with softmax for multi-class classification.
    """
    # Pad the input with zeros to make it a square matrix
    input_tensor_padded = F.pad(input_tensor, (0, input_tensor.shape[1] - input_tensor.shape[0]))
    
    # Compute the diagonal elements
    diagonal_elements = torch.diagflat(input_tensor_padded)
    
    # Calculate the log softmax probabilities
    log_softmax_probs = F.log_softmax(input_tensor_padded, dim=1)
    
    # Compute the multi-label margin loss
    loss = F.multilabel_margin_loss(log_softmax_probs, diagonal_elements, reduction="mean")
    
    return loss

function_signature = {
    "name": "torch_multi_label_margin_loss_with_softmax",
    "inputs": [
        ((5, 5), torch.float32),
        ((5,), torch.long)
    ],
    "outputs": [
        ((), torch.float32)
    ]
}
