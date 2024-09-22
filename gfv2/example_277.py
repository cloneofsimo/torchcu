
import torch

def soft_margin_loss_with_attention(input_tensor: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Calculates the soft margin loss with global attention. 
    """
    
    # Ensure attention weights are in the correct shape
    attention_weights = attention_weights.view(attention_weights.size(0), 1, -1)
    
    # Apply attention to the input tensor
    attended_input = input_tensor * attention_weights
    
    # Calculate the soft margin loss
    loss = torch.nn.functional.soft_margin_loss(attended_input, torch.ones_like(attended_input), reduction='none')
    
    # Return the loss tensor
    return loss

function_signature = {
    "name": "soft_margin_loss_with_attention",
    "inputs": [
        ((10, 10), torch.float32),
        ((10,), torch.float32)
    ],
    "outputs": [
        ((10, 10), torch.float32),
    ]
}
