
import torch
import torch.nn.functional as F


def masked_attention_transposed_conv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                      mask: torch.Tensor,
                                      weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs masked attention followed by a transposed convolution with bias.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, query_len, hidden_dim).
        key (torch.Tensor): Key tensor of shape (batch_size, key_len, hidden_dim).
        value (torch.Tensor): Value tensor of shape (batch_size, key_len, hidden_dim).
        mask (torch.Tensor): Mask tensor of shape (batch_size, query_len, key_len).
        weight (torch.Tensor): Weight tensor for transposed convolution of shape (out_channels, in_channels, kernel_size).
        bias (torch.Tensor): Bias tensor for transposed convolution of shape (out_channels).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, output_len).
    """
    # Masked attention
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    attention_scores = attention_scores / (query.size(-1) ** 0.5)
    attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(attention_scores, dim=-1)
    context = torch.matmul(attention_weights, value)

    # Transposed convolution
    output = F.conv_transpose1d(context, weight, bias=bias)

    return output

function_signature = {
    "name": "masked_attention_transposed_conv",
    "inputs": [
        ((1, 10, 512), torch.float32),
        ((1, 20, 512), torch.float32),
        ((1, 20, 512), torch.float32),
        ((1, 10, 20), torch.bool),
        ((256, 512, 3), torch.float32),
        ((256,), torch.float32)
    ],
    "outputs": [
        ((1, 256, 12), torch.float32),
    ]
}
