
import torch
import torch.nn as nn

def multi_label_margin_loss_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Computes the multi-label margin loss for a given input and target tensor.
    The loss encourages the input to be closer to the target label and further
    away from other labels by a specified margin.
    """
    
    # Get the batch size and number of classes
    batch_size, num_classes = input_tensor.size()

    # Calculate the positive term
    positive_term = (1 - input_tensor * target_tensor).clamp(min=0)

    # Calculate the negative term
    negative_term = (input_tensor * (1 - target_tensor) + margin).clamp(min=0)

    # Calculate the loss for each sample
    loss = (positive_term.sum(dim=1) + negative_term.sum(dim=1)) / num_classes

    return loss

function_signature = {
    "name": "multi_label_margin_loss_function",
    "inputs": [
        ((10, 5), torch.float32),
        ((10, 5), torch.float32)
    ],
    "outputs": [
        ((10,), torch.float32),
    ]
}
