### 1. `sample_from_multinomial`
```python
import torch

def sample_from_multinomial(logits: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Samples from a multinomial distribution.

    Args:
        logits (torch.Tensor): Logits of the multinomial distribution.
        num_samples (int): Number of samples to draw.

    Returns:
        torch.Tensor: A tensor of shape `(num_samples, logits.shape[-1])` containing the samples.
    """
    return torch.multinomial(logits.softmax(dim=-1), num_samples=num_samples)
```

### 2. `forward_with_mask`
```python
import torch

def forward_with_mask(inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a mask to the inputs before passing them through a forward pass.

    Args:
        inputs (torch.Tensor): The input tensor.
        attention_mask (torch.Tensor): The attention mask tensor.

    Returns:
        torch.Tensor: The masked input tensor.
    """
    return inputs * attention_mask.unsqueeze(-1)
```

### 3. `compute_attention_weights`
```python
import torch

def compute_attention_weights(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the attention weights using the query, key, and value tensors.

    Args:
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
        value (torch.Tensor): The value tensor.
        attention_mask (torch.Tensor): The attention mask tensor.

    Returns:
        torch.Tensor: The attention weights tensor.
    """
    attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (key.shape[-1] ** 0.5)
    attention_scores = attention_scores + attention_mask.unsqueeze(-1)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    return attention_weights
```