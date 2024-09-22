
import torch
import torch.nn.functional as F

def multi_scale_attention_function(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs multi-scale attention with attention mask.
    """
    batch_size, seq_len, embedding_dim = query.size()
    
    # Create a list of scales to attend to (e.g., [1, 2, 4])
    scales = [1, 2, 4]
    
    # Initialize the output tensor
    output = torch.zeros_like(query)
    
    for scale in scales:
        # Resize the query, key, and value tensors for the current scale
        query_scaled = F.interpolate(query, size=seq_len // scale, mode='linear', align_corners=False)
        key_scaled = F.interpolate(key, size=seq_len // scale, mode='linear', align_corners=False)
        value_scaled = F.interpolate(value, size=seq_len // scale, mode='linear', align_corners=False)
        
        # Calculate the attention scores
        attention_scores = torch.matmul(query_scaled, key_scaled.transpose(-2, -1)) / math.sqrt(embedding_dim)
        
        # Apply the attention mask
        attention_scores = attention_scores.masked_fill(attention_mask[:, :seq_len // scale, :seq_len // scale] == 0, float('-inf'))
        
        # Calculate the attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Calculate the context vector
        context_vector = torch.matmul(attention_weights, value_scaled)
        
        # Upsample the context vector back to the original size
        context_vector = F.interpolate(context_vector, size=seq_len, mode='linear', align_corners=False)
        
        # Add the context vector to the output
        output += context_vector
    
    return output / len(scales)


function_signature = {
    "name": "multi_scale_attention_function",
    "inputs": [
        ((8, 128, 512), torch.float32),
        ((8, 128, 512), torch.float32),
        ((8, 128, 512), torch.float32),
        ((8, 128, 128), torch.bool),  
    ],
    "outputs": [
        ((8, 128, 512), torch.float32),
    ]
}
