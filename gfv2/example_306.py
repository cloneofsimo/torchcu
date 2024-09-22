
import torch
import torch.nn.functional as F

def ctc_attention_loss_fp16(input_tensor: torch.Tensor, target_tensor: torch.Tensor, 
                             attention_weights: torch.Tensor, blank_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates CTC loss with windowed attention mechanism using fp16 precision.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape (batch_size, seq_len, vocab_size) in fp32.
        target_tensor (torch.Tensor): Target tensor with shape (batch_size, target_len) in int64.
        attention_weights (torch.Tensor): Attention weights with shape (batch_size, seq_len) in fp32.
        blank_id (int, optional): Blank label ID. Defaults to 0.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Loss value (float)
            - Gradients of the input tensor (torch.Tensor)
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    attention_weights_bf16 = attention_weights.to(torch.bfloat16)
    
    # Apply windowed attention
    input_bf16 = input_bf16 * attention_weights_bf16.unsqueeze(dim=-1)

    # Calculate CTC loss
    loss = F.ctc_loss(input_bf16.to(torch.float32), target_tensor, input_lengths=torch.full((input_tensor.size(0),), input_tensor.size(1), dtype=torch.long),
                     target_lengths=torch.full((target_tensor.size(0),), target_tensor.size(1), dtype=torch.long), 
                     blank=blank_id, reduction='mean')

    # Calculate gradients
    loss.backward()

    # Return loss and gradients
    return loss.item(), input_tensor.grad.to(torch.float32)

function_signature = {
    "name": "ctc_attention_loss_fp16",
    "inputs": [
        ((1, 10, 50), torch.float32),
        ((1, 10), torch.int64),
        ((1, 10), torch.float32),
    ],
    "outputs": [
        ((), torch.float32),
        ((1, 10, 50), torch.float32),
    ]
}
