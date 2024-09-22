
import torch
import torch.nn as nn

def multi_loss_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """
    Calculates a multi-loss combination for a classification task: MSE, sigmoid focal loss, and cross-entropy. 
    This function is for demonstration, and you should use the appropriate loss function based on your problem. 

    Args:
        input_tensor: The model output, typically a batch of logits.
        target_tensor: The target labels for the classification task.
        class_weights: Weights for each class, used for balancing the loss.

    Returns:
        A single loss tensor representing the combination of the three losses.
    """
    
    # Flatten the input and target tensors (assuming they are 2D, Batch x Features)
    input_tensor = input_tensor.flatten(start_dim=1)
    target_tensor = target_tensor.flatten(start_dim=1)
    
    # MSE Loss
    mse_loss = nn.functional.mse_loss(input_tensor, target_tensor, reduction='mean')
    
    # Sigmoid Focal Loss
    sigmoid_focal_loss = nn.functional.binary_cross_entropy_with_logits(
        input_tensor, target_tensor, pos_weight=class_weights, reduction='mean'
    )
    
    # Cross-Entropy Loss
    cross_entropy_loss = nn.functional.cross_entropy(input_tensor, target_tensor, weight=class_weights)
    
    # Combine losses (adjust weights as needed for your application)
    combined_loss = mse_loss + sigmoid_focal_loss + cross_entropy_loss
    
    return combined_loss

function_signature = {
    "name": "multi_loss_function",
    "inputs": [
        ((1, 10), torch.float32),
        ((1, 10), torch.long),
        ((10,), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}

