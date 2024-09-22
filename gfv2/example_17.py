
import torch
import torch.nn.functional as F

def torch_cosine_embedding_rms_loss_fp16(input1: torch.Tensor, input2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine embedding loss with root mean square energy (RMSE) normalization
    and uses fp16 precision for intermediate calculations.
    
    Args:
        input1 (torch.Tensor): First input tensor with shape (batch_size, embedding_dim).
        input2 (torch.Tensor): Second input tensor with shape (batch_size, embedding_dim).
        labels (torch.Tensor): Target labels with shape (batch_size,).
        
    Returns:
        torch.Tensor: The cosine embedding loss with RMSE normalization.
    """
    input1_fp16 = input1.to(torch.float16)
    input2_fp16 = input2.to(torch.float16)
    
    # Compute RMSE normalization
    rmse1 = torch.sqrt(torch.mean(input1_fp16 * input1_fp16, dim=1, keepdim=True))
    rmse2 = torch.sqrt(torch.mean(input2_fp16 * input2_fp16, dim=1, keepdim=True))

    # Normalize inputs
    input1_normalized = input1_fp16 / rmse1
    input2_normalized = input2_fp16 / rmse2
    
    # Compute cosine similarity
    cosine_similarity = F.cosine_similarity(input1_normalized, input2_normalized, dim=1)
    
    # Calculate cosine embedding loss
    loss = F.cosine_embedding_loss(input1_normalized, input2_normalized, labels)
    
    return loss.to(torch.float32)

function_signature = {
    "name": "torch_cosine_embedding_rms_loss_fp16",
    "inputs": [
        ((4, 128), torch.float32),  # Shape of input1
        ((4, 128), torch.float32),  # Shape of input2
        ((4,), torch.int64)       # Shape of labels
    ],
    "outputs": [
        ((), torch.float32),  # Shape of the loss scalar
    ]
}
