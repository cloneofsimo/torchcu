
import torch
import torch.nn.functional as F

def multi_scale_attention_func(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                                 scales: list[int]) -> torch.Tensor:
  """
  Multi-scale attention mechanism.

  Args:
    query: Query tensor of shape (B, N, D).
    key: Key tensor of shape (B, N, D).
    value: Value tensor of shape (B, N, D).
    scales: List of scales for multi-scale attention.

  Returns:
    Output tensor of shape (B, N, D).
  """
  B, N, D = query.shape
  outputs = []
  for scale in scales:
    # Apply downsampling to query, key, and value
    downsampled_query = F.avg_pool1d(query.transpose(1, 2), scale).transpose(1, 2)
    downsampled_key = F.avg_pool1d(key.transpose(1, 2), scale).transpose(1, 2)
    downsampled_value = F.avg_pool1d(value.transpose(1, 2), scale).transpose(1, 2)

    # Calculate attention weights
    attention_weights = torch.matmul(downsampled_query, downsampled_key.transpose(1, 2))
    attention_weights = F.softmax(attention_weights, dim=-1)

    # Apply attention weights to value
    output = torch.matmul(attention_weights, downsampled_value)
    outputs.append(output)

  # Concatenate outputs from different scales
  output = torch.cat(outputs, dim=-1)
  return output

function_signature = {
    "name": "multi_scale_attention_func",
    "inputs": [
        ((16, 32, 128), torch.float32),
        ((16, 32, 128), torch.float32),
        ((16, 32, 128), torch.float32),
        ([2, 4], torch.int32)
    ],
    "outputs": [
        ((16, 32, 128), torch.float32),
    ]
}
